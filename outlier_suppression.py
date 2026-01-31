"""
Outlier Suppression Module

Detects and suppresses outlier tokens based on attention mechanism.
Identifies top-k tokens with highest Attn[cls,i] / Attn[i,i] ratio as outliers,
and replaces them with weighted mean of their spatial neighbors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


def detect_outliers_by_attention(attn_weights: torch.Tensor, num_patches: int, top_k: int = 10) -> torch.Tensor:
    """
    Detect top-k outliers based on attention mechanism:
    Select top k tokens with largest Attn[cls,i] / Attn[i,i] ratio.
    
    Args:
        attn_weights: attention weights of shape [batch_size, num_heads, seq_len, seq_len]
                     or [batch_size, seq_len, seq_len]
        num_patches: number of patch tokens (excluding cls token)
        top_k: number of top outliers to select (default: 10)
    
    Returns:
        outlier_indices: indices of top-k outlier tokens [batch_size, top_k] or [top_k] if single batch
    """
    # Handle different input shapes
    if len(attn_weights.shape) == 4:
        # Shape is [batch_size, num_heads, seq_len, seq_len], average over heads
        attn_weights = attn_weights.mean(dim=1)  # [batch_size, seq_len, seq_len]
    
    if len(attn_weights.shape) == 3:
        # Shape is [batch_size, seq_len, seq_len]
        batch_size = attn_weights.shape[0]
        single_batch = False
    else:
        # Single sample [seq_len, seq_len]
        attn_weights = attn_weights.unsqueeze(0)
        batch_size = 1
        single_batch = True
    
    # Get self-attention values (diagonal elements, excluding cls token)
    # attn_weights: [batch_size, seq_len, seq_len]
    self_attn = torch.stack([attn_weights[b].diag()[1:1+num_patches] for b in range(batch_size)])  # [batch_size, num_patches]
    
    # Get cls-to-token attention (first row, excluding cls itself)
    cls_to_token_attn = attn_weights[:, 0, 1:1+num_patches]  # [batch_size, num_patches]
    
    # Compute ratio: Attn[cls,i] / Attn[i,i]
    # Add small epsilon to avoid division by zero
    ratio = cls_to_token_attn / (self_attn + 1e-8)  # [batch_size, num_patches]
    
    # Select top-k tokens with largest ratios
    actual_k = min(top_k, num_patches)
    _, outlier_indices = torch.topk(ratio, k=actual_k, largest=True, dim=1)  # [batch_size, top_k]
    
    if single_batch:
        return outlier_indices[0]  # Return [top_k] for single batch
    return outlier_indices


class OutlierSuppressionModule(nn.Module):
    """
    Suppresses outlier tokens and removes their contamination from neighbors.
    
    Bidirectional decontamination:
    1. Replace outlier with weighted mean of neighbors
    2. Remove \u03c3 * x_outlier from each neighbor, where \u03c3 = similarity * temp
    
    Args:
        top_k: number of outlier tokens to suppress per image (default: 10)
        contamination_temp: temperature for contamination removal (default: 0.5)
                          Higher = milder removal, Lower = aggressive removal
    """
    
    def __init__(self, top_k: int = 10, contamination_temp: float = 0.1):
        super().__init__()
        self.top_k = top_k
        self.contamination_temp = contamination_temp
    
    def forward(
        self,
        feature_map: torch.Tensor,
        attn_weights: torch.Tensor,
        grid_h: int,
        grid_w: int
    ) -> torch.Tensor:
        """
        Apply outlier suppression to feature map.
        
        Args:
            feature_map: patch features [B, C, H, W]
            attn_weights: attention weights from transformer block [B, num_heads, N, N] or [B, N, N]
            grid_h, grid_w: spatial grid dimensions
            
        Returns:
            suppressed_feature_map: [B, C, H, W]
        """
        B, C, H, W = feature_map.shape
        num_patches = H * W
        
        # Detect outliers
        outlier_indices = detect_outliers_by_attention(attn_weights, num_patches, self.top_k)  # [B, top_k]
        
        # Convert flat indices to 2D coordinates
        if len(outlier_indices.shape) == 1:
            # Single batch
            outlier_indices = outlier_indices.unsqueeze(0)
        
        # Apply mean interpolation
        return self.mean_interpolation(feature_map, outlier_indices, grid_h, grid_w)
    
    def mean_interpolation(
        self,
        feature_map: torch.Tensor,
        outlier_indices: torch.Tensor,
        grid_h: int,
        grid_w: int
    ) -> torch.Tensor:
        """
        Replace outlier features with weighted average of their 8 neighbors.
        Additionally, removes contamination from neighbors caused by the outlier.
        
        Args:
            feature_map: [B, C, H, W]
            outlier_indices: flat indices of outliers [B, num_outliers]
            grid_h, grid_w: spatial dimensions
            
        Returns:
            feature_map with outliers replaced and neighbors decontaminated [B, C, H, W]
        """
        B, C, H, W = feature_map.shape
        device = feature_map.device
        
        if outlier_indices.numel() == 0:
            return feature_map
        
        result = feature_map.clone()
        
        # Process each batch
        for b in range(B):
            batch_outliers = outlier_indices[b]  # [num_outliers]
            
            if batch_outliers.numel() == 0:
                continue
            
            # Convert flat indices to 2D coordinates
            outlier_rows = torch.div(batch_outliers, grid_w, rounding_mode='trunc')
            outlier_cols = batch_outliers % grid_w
            outlier_coords = torch.stack([outlier_rows, outlier_cols], dim=1)  # [num_outliers, 2]
            
            num_outliers = outlier_coords.shape[0]
            
            # Create neighbor offsets (8 neighbors)
            offsets = torch.tensor([[-1, -1], [-1, 0], [-1, 1],
                                    [0, -1],          [0, 1],
                                    [1, -1],  [1, 0],  [1, 1]], 
                                   dtype=torch.long, device=device)  # [8, 2]
            
            # Broadcast to get all neighbor coordinates: [num_outliers, 8, 2]
            neighbor_coords = outlier_coords.unsqueeze(1) + offsets.unsqueeze(0)
            
            # Clamp coordinates to valid range
            neighbor_coords[..., 0] = torch.clamp(neighbor_coords[..., 0], 0, H - 1)
            neighbor_coords[..., 1] = torch.clamp(neighbor_coords[..., 1], 0, W - 1)
            
            # Get outlier features: [num_outliers, C]
            outlier_feats = feature_map[b, :, outlier_coords[:, 0], outlier_coords[:, 1]].T  # [num_outliers, C]
            
            # Get neighbor features: [num_outliers, 8, C]
            neighbor_feats = feature_map[b, :, neighbor_coords[..., 0], neighbor_coords[..., 1]].permute(1, 2, 0)
            
            # Normalize for cosine similarity
            outlier_norm = F.normalize(outlier_feats, p=2, dim=1)  # [num_outliers, C]
            neighbor_norm = F.normalize(neighbor_feats, p=2, dim=2)  # [num_outliers, 8, C]
            
            # Compute cosine similarity: [num_outliers, 8]
            similarity = (neighbor_norm * outlier_norm.unsqueeze(1)).sum(dim=2)
            
            # Inverse similarity as weights for replacement (more different = higher weight)
            weights = 1.0 - similarity  # [num_outliers, 8]
            weights = torch.clamp(weights, min=0.0)
            weights = F.softmax(weights, dim=1)  # [num_outliers, 8]
            
            # Compute weighted average for outlier replacement: [num_outliers, C]
            weighted_avg = (neighbor_feats * weights.unsqueeze(2)).sum(dim=1)
            
            # ===== BIDIRECTIONAL DECONTAMINATION =====
            # Remove contamination from neighbors: neighbor_clean = neighbor - σ * outlier
            # where σ = similarity * temperature (scaled for mild removal)
            
            # Temperature-scaled similarity for contamination removal
            contamination_strength = similarity * self.contamination_temp  # [num_outliers, 8]
            contamination_strength = torch.clamp(contamination_strength, 0, 1)  # Ensure valid range
            
            # For each outlier-neighbor pair, remove: σ * x_outlier from neighbor
            # neighbor_decontaminated = neighbor - σ * outlier
            outlier_contamination = outlier_feats.unsqueeze(1) * contamination_strength.unsqueeze(2)  # [num_outliers, 8, C]
            neighbor_feats_clean = neighbor_feats - outlier_contamination
            
            # Update neighbor positions with decontaminated features
            for i in range(num_outliers):
                for j in range(8):
                    ny, nx = neighbor_coords[i, j, 0].item(), neighbor_coords[i, j, 1].item()
                    # Only update if it's a valid neighbor position (not clamped to same outlier)
                    if ny != outlier_coords[i, 0].item() or nx != outlier_coords[i, 1].item():
                        result[b, :, ny, nx] = neighbor_feats_clean[i, j]
            
            # Update outlier positions with weighted average
            result[b, :, outlier_coords[:, 0], outlier_coords[:, 1]] = weighted_avg.T
        
        return result
