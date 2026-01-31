"""
Self-Attention Enhancement Module

Enhances tokens' self-attention to make them more robust and self-reliant.
Tokens with weak self-attention are encouraged to attend more to themselves,
reducing their dependency on other tokens and making them less prone to outlier behavior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SelfAttentionEnhancementModule(nn.Module):
    """
    Enhances self-attention for tokens with weak self-attention.
    
    Two strategies:
    1. Feature-based: Strengthen features based on self-attention deficit
    2. Attention-based: Re-weight attention maps to boost self-attention
    
    Args:
        enhancement_strength: Controls how much to enhance self-attention (default: 0.1)
                            Higher = stronger enhancement
        min_self_attn_threshold: Minimum desired self-attention value (default: 0.15)
                                Tokens below this get enhanced
        mode: 'feature' or 'attention' - which enhancement strategy to use
        top_k: Number of tokens with lowest self-attention to enhance (default: 10)
               If None, uses threshold-based enhancement
    """
    
    def __init__(
        self, 
        enhancement_strength: float = 0.1,
        min_self_attn_threshold: float = 0.15,
        mode: str = 'feature',
        top_k: int = 10
    ):
        super().__init__()
        self.enhancement_strength = enhancement_strength
        self.min_self_attn_threshold = min_self_attn_threshold
        self.mode = mode
        self.top_k = top_k
        
        assert mode in ['feature', 'attention'], f"Mode must be 'feature' or 'attention', got {mode}"
        
    def forward(
        self, 
        features: torch.Tensor,
        attn_weights: torch.Tensor,
        grid_h: int,
        grid_w: int
    ) -> torch.Tensor:
        """
        Apply self-attention enhancement.
        
        Args:
            features: patch features [B, C, H, W] or [B, N, C]
            attn_weights: attention weights [B, num_heads, N, N] or [B, N, N]
            grid_h, grid_w: spatial grid dimensions
            
        Returns:
            enhanced_features: [B, C, H, W] or [B, N, C] (same shape as input)
        """
        if self.mode == 'feature':
            return self.enhance_features(features, attn_weights, grid_h, grid_w)
        else:
            return self.enhance_via_attention(features, attn_weights, grid_h, grid_w)
    
    def enhance_features(
        self,
        features: torch.Tensor,
        attn_weights: torch.Tensor,
        grid_h: int,
        grid_w: int
    ) -> torch.Tensor:
        """
        Enhance features for tokens with weak self-attention.
        
        Strategy: Select top-k tokens with lowest self-attention and replace them
        with the weighted mean of their 8 spatial neighbors.
        
        Args:
            features: [B, C, H, W] (patch tokens only, no CLS) or [B, N, C] (with CLS)
            attn_weights: [B, num_heads, N, N] or [B, N, N] (includes CLS)
            
        Returns:
            enhanced_features: same shape as input
        """
        input_shape = features.shape
        is_spatial = len(input_shape) == 4
        
        if is_spatial:
            # Convert [B, C, H, W] to [B, N, C] - these are patch tokens only
            B, C, H, W = features.shape
            features_seq = features.view(B, C, H * W).permute(0, 2, 1)  # [B, num_patches, C]
            has_cls = False
        else:
            B, N, C = features.shape
            features_seq = features
            # Check if features include CLS token by comparing with attention size
            has_cls = (N == attn_weights.shape[1])  # If same as attn, includes CLS
        
        # Average attention over heads if needed
        if len(attn_weights.shape) == 4:
            attn_weights = attn_weights.mean(dim=1)  # [B, N, N] where N includes CLS
        
        num_patches = attn_weights.shape[1] - 1  # Exclude CLS token from attention
        
        # Extract self-attention values (diagonal) for patch tokens only
        # attn_weights[:, i, i] = self-attention of token i
        self_attn = torch.stack([attn_weights[b].diag()[1:1+num_patches] for b in range(B)])  # [B, num_patches]
        
        # Select top-k tokens with lowest self-attention
        actual_k = min(self.top_k, num_patches)
        _, weak_indices = torch.topk(self_attn, k=actual_k, largest=False, dim=1)  # [B, top_k] - lowest values
        
        # Extract patch features (skip CLS if present)
        if has_cls:
            cls_token = features_seq[:, 0:1, :]  # [B, 1, C]
            patch_features = features_seq[:, 1:, :]  # [B, num_patches, C]
        else:
            patch_features = features_seq  # Already patch-only [B, num_patches, C]
        
        # Convert to spatial format for neighbor computation
        patch_features_spatial = patch_features.permute(0, 2, 1).view(B, C, grid_h, grid_w)  # [B, C, H, W]
        
        # Apply neighbor mean replacement for weak tokens
        enhanced_patches_spatial = self.replace_weak_tokens_with_neighbors(
            patch_features_spatial, weak_indices, grid_h, grid_w
        )
        
        # Convert back to sequence format
        enhanced_patches = enhanced_patches_spatial.view(B, C, grid_h * grid_w).permute(0, 2, 1)  # [B, num_patches, C]
        
        # Reconstruct with CLS token if it was present
        if has_cls:
            enhanced_features = torch.cat([cls_token, enhanced_patches], dim=1)  # [B, N, C]
        else:
            enhanced_features = enhanced_patches  # [B, num_patches, C]
        
        # Convert back to spatial format if needed
        if is_spatial:
            if has_cls:
                # Remove CLS for spatial format
                enhanced_features = enhanced_features[:, 1:, :]
            enhanced_features = enhanced_features.permute(0, 2, 1).view(B, C, H, W)
        
        return enhanced_features
    
    def enhance_via_attention(
        self,
        features: torch.Tensor,
        attn_weights: torch.Tensor,
        grid_h: int,
        grid_w: int
    ) -> torch.Tensor:
        """
        Enhance features by modifying attention weights to boost self-attention.
        
        Strategy: Re-weight attention map to increase diagonal (self-attention) values,
        then use modified attention to re-compute features.
        
        Args:
            features: [B, C, H, W] (patch tokens only, no CLS) or [B, N, C] (with CLS)
            attn_weights: [B, num_heads, N, N] or [B, N, N] (includes CLS)
            
        Returns:
            enhanced_features: same shape as input
        """
        input_shape = features.shape
        is_spatial = len(input_shape) == 4
        
        if is_spatial:
            B, C, H, W = features.shape
            features_seq = features.view(B, C, H * W).permute(0, 2, 1)  # [B, num_patches, C]
            has_cls = False
        else:
            B, N, C = features.shape
            features_seq = features
            # Check if features include CLS token
            has_cls = (N == attn_weights.shape[1])
        
        # Average attention over heads if needed
        if len(attn_weights.shape) == 4:
            attn_avg = attn_weights.mean(dim=1)  # [B, N, N]
        else:
            attn_avg = attn_weights
        
        N = attn_avg.shape[1]
        num_patches = N - 1
        
        # Extract self-attention values for patches
        self_attn = torch.stack([attn_avg[b].diag()[1:1+num_patches] for b in range(B)])  # [B, num_patches]
        
        # Compute boost factor for weak self-attention tokens
        boost_factor = torch.clamp(
            self.min_self_attn_threshold - self_attn,
            min=0.0
        ) * self.enhancement_strength  # [B, num_patches]
        
        # Create modified attention weights with boosted diagonal
        attn_modified = attn_avg.clone()
        
        for b in range(B):
            for i in range(num_patches):
                patch_idx = i + 1  # +1 for CLS token
                # Boost self-attention (diagonal)
                attn_modified[b, patch_idx, patch_idx] += boost_factor[b, i]
        
        # Re-normalize attention (each row should sum to 1)
        attn_modified = attn_modified / (attn_modified.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Prepare features for attention aggregation
        if has_cls:
            # Features include CLS, use them directly
            features_for_attn = features_seq  # [B, N, C]
        else:
            # Features are patch-only, need to add dummy CLS for attention computation
            # We'll only use the patch outputs anyway
            dummy_cls = torch.zeros(B, 1, C, device=features_seq.device, dtype=features_seq.dtype)
            features_for_attn = torch.cat([dummy_cls, features_seq], dim=1)  # [B, N, C]
        
        # Apply modified attention to aggregate features
        # For patch tokens: enhanced[i] = sum_j(attn_modified[i,j] * features[j])
        enhanced_features_seq = torch.bmm(attn_modified, features_for_attn)  # [B, N, C]
        
        # Extract the appropriate output
        if has_cls:
            output_seq = enhanced_features_seq
        else:
            # Remove dummy CLS, keep only patches
            output_seq = enhanced_features_seq[:, 1:, :]  # [B, num_patches, C]
        
        # Convert back to spatial format if needed
        if is_spatial:
            if has_cls:
                # Remove CLS for spatial format
                output_seq = output_seq[:, 1:, :]
            enhanced_features = output_seq.permute(0, 2, 1).view(B, C, H, W)
        else:
            enhanced_features = output_seq
        
        return enhanced_features
    
    def replace_weak_tokens_with_neighbors(
        self,
        feature_map: torch.Tensor,
        weak_indices: torch.Tensor,
        grid_h: int,
        grid_w: int
    ) -> torch.Tensor:
        """
        Replace weak tokens with weighted average of their 8 spatial neighbors.
        
        Args:
            feature_map: [B, C, H, W] patch features in spatial format
            weak_indices: flat indices of weak tokens [B, num_weak]
            grid_h, grid_w: spatial dimensions
            
        Returns:
            feature_map with weak tokens replaced by neighbor means [B, C, H, W]
        """
        B, C, H, W = feature_map.shape
        device = feature_map.device
        
        if weak_indices.numel() == 0:
            return feature_map
        
        result = feature_map.clone()
        
        # Process each batch
        for b in range(B):
            batch_weak = weak_indices[b]  # [num_weak]
            
            if batch_weak.numel() == 0:
                continue
            
            # Convert flat indices to 2D coordinates
            weak_rows = torch.div(batch_weak, grid_w, rounding_mode='trunc')
            weak_cols = batch_weak % grid_w
            weak_coords = torch.stack([weak_rows, weak_cols], dim=1)  # [num_weak, 2]
            
            num_weak = weak_coords.shape[0]
            
            # Create neighbor offsets (8 neighbors)
            offsets = torch.tensor([[-1, -1], [-1, 0], [-1, 1],
                                    [0, -1],          [0, 1],
                                    [1, -1],  [1, 0],  [1, 1]], 
                                   dtype=torch.long, device=device)  # [8, 2]
            
            # Broadcast to get all neighbor coordinates: [num_weak, 8, 2]
            neighbor_coords = weak_coords.unsqueeze(1) + offsets.unsqueeze(0)
            
            # Clamp coordinates to valid range
            neighbor_coords[..., 0] = torch.clamp(neighbor_coords[..., 0], 0, H - 1)
            neighbor_coords[..., 1] = torch.clamp(neighbor_coords[..., 1], 0, W - 1)
            
            # Get weak token features: [num_weak, C]
            weak_feats = feature_map[b, :, weak_coords[:, 0], weak_coords[:, 1]].T  # [num_weak, C]
            
            # Get neighbor features: [num_weak, 8, C]
            neighbor_feats = feature_map[b, :, neighbor_coords[..., 0], neighbor_coords[..., 1]].permute(1, 2, 0)
            
            # Normalize for cosine similarity
            weak_norm = F.normalize(weak_feats, p=2, dim=1)  # [num_weak, C]
            neighbor_norm = F.normalize(neighbor_feats, p=2, dim=2)  # [num_weak, 8, C]
            
            # Compute cosine similarity: [num_weak, 8]
            similarity = (neighbor_norm * weak_norm.unsqueeze(1)).sum(dim=2)
            
            # Inverse similarity as weights for replacement (more different = higher weight)
            weights = 1.0 - similarity  # [num_weak, 8]
            weights = torch.clamp(weights, min=0.0)
            weights = F.softmax(weights, dim=1)  # [num_weak, 8]
            
            # Compute weighted average for weak token replacement: [num_weak, C]
            weighted_avg = (neighbor_feats * weights.unsqueeze(2)).sum(dim=1)
            
            # Update weak token positions with weighted average
            result[b, :, weak_coords[:, 0], weak_coords[:, 1]] = weighted_avg.T
        
        return result
    
    def get_self_attention_stats(self, attn_weights: torch.Tensor) -> dict:
        """
        Compute statistics about self-attention values.
        Useful for monitoring and debugging.
        
        Args:
            attn_weights: [B, num_heads, N, N] or [B, N, N]
            
        Returns:
            dict with statistics
        """
        # Average over heads if needed
        if len(attn_weights.shape) == 4:
            attn_avg = attn_weights.mean(dim=1)
        else:
            attn_avg = attn_weights
        
        B, N, _ = attn_avg.shape
        num_patches = N - 1
        
        # Extract self-attention values for patches
        self_attn = torch.stack([attn_avg[b].diag()[1:1+num_patches] for b in range(B)])
        
        return {
            'mean': self_attn.mean().item(),
            'std': self_attn.std().item(),
            'min': self_attn.min().item(),
            'max': self_attn.max().item(),
            'below_threshold': (self_attn < self.min_self_attn_threshold).float().mean().item()
        }
