"""
Similarity-based Attention Enhancement Module

Enhances attention weights by adding self-similarity map computed from mid-layer features.
The self-similarity map captures patch-to-patch semantic relationships, which when added
to attention weights, helps emphasize salient features and improve semantic coherence.

This is a completely training-free module with no learnable parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimilarityEnhancementModule(nn.Module):
    """
    Enhances attention weights by adding self-similarity map from mid-layer features.
    
    The similarity map M[i,j] = CosSim(x_i, x_j) captures semantic relationships between
    patches. Adding this to attention weights emphasizes connections between semantically
    similar regions, enhancing salient features.
    
    Args:
        similarity_weight: Weight for similarity map when adding to attention (default: 1.0)
        temperature: Temperature for similarity computation (default: 1.0)
    """
    
    def __init__(self, similarity_weight=1.0, temperature=1.0, add_self_similarity=True):
        super().__init__()
        self.similarity_weight = similarity_weight
        self.temperature = temperature
        self.add_self_similarity = add_self_similarity
        # Cache for similarity map to be used during attention computation
        self.cached_similarity_map = None
    
    def compute_similarity_map(self, features, scale=None):
        """
        Compute pairwise cosine similarity map.
        
        Args:
            features: [B, N, D] patch features (BND format expected)
            scale: Optional scaling factor (like attention scale)
            
        Returns:
            similarity_map: [B, N, N] pairwise cosine similarities (scaled)
        """
        # Input should be in BND format [batch, num_patches, dim]
        # No format conversion needed - caller should provide correct format
        
        # Normalize features for cosine similarity
        features_norm = F.normalize(features.float(), p=2, dim=-1)  # [B, N, D]
        
        # Compute similarity matrix: M[i,j] = CosSim(x_i, x_j)
        similarity_map = torch.bmm(features_norm, features_norm.transpose(1, 2))  # [B, N, N]
        
        # Apply temperature scaling
        similarity_map = similarity_map / self.temperature
        
        # Optionally remove self-similarity (diagonal)
        if not self.add_self_similarity:
            B, N = similarity_map.shape[0], similarity_map.shape[1]
            mask = torch.eye(N, device=similarity_map.device, dtype=similarity_map.dtype).unsqueeze(0)
            similarity_map = similarity_map * (1 - mask)
        
        return similarity_map
    
    def cache_similarity_map(self, mid_features):
        """
        Cache the similarity map computed from mid-layer features.
        This should be called before the custom_attn computation.
        
        Args:
            mid_features: [B, N, D] features from mid layer (patch features, excluding CLS)
        """
        self.cached_similarity_map = self.compute_similarity_map(mid_features)
    
    def enhance_attention_logits(self, attn_logits, num_heads=None):
        """
        Enhance attention logits (pre-softmax) using the cached similarity map.
        
        This aligns with SegEarth/SC-CLIP/SFP approach of modifying the logits sum directly,
        which is more numerically stable and preserves distribution properties better than
        averaging probabilities.
        
        Args:
            attn_logits: [B*num_heads, N, N] attention logits
            num_heads: Number of attention heads
            
        Returns:
            attn_logits: [B*num_heads, N, N] Enhanced attention logits
        """
        if self.cached_similarity_map is None:
            return attn_logits
            
        sim_map = self.cached_similarity_map.clone()  # [B, num_patches, num_patches]
        B, num_patches, _ = sim_map.shape
        
        # Determine if we need to pad for CLS token
        # attn_logits is [B*num_heads, N_attn, N_attn]
        N_attn = attn_logits.shape[-1]
        
        # Prepare Sim Map
        # 1. Expand to heads
        if num_heads is not None:
            sim_map = sim_map.unsqueeze(1).repeat(1, num_heads, 1, 1) # [B, num_heads, Np, Np]
            sim_map = sim_map.view(-1, num_patches, num_patches)      # [B*num_heads, Np, Np]
        
        # 2. Pad to match N_attn (handle CLS token)
        if N_attn == num_patches + 1:
            # Create padded map with 0s for CLS interactions
            padded_sim = torch.zeros(sim_map.shape[0], N_attn, N_attn, 
                                     device=sim_map.device, dtype=sim_map.dtype)
            
            # Place similarity map in bottom-right (Patch-Patch interactions)
            padded_sim[:, 1:, 1:] = sim_map
            
            # For CLS-Patch and Patch-CLS, we leave as 0
            # This means we don't bias these interactions, letting original attention decide
            sim_map = padded_sim
            
        elif N_attn != num_patches:
            # Mismatch size, return original
            return attn_logits

        # Apply Enhancement
        # SFP/SegEarth Logic: combined_logits = original_logits + lambda * sim_map
        # Using similarity_weight parameter
        sim_map = sim_map.to(attn_logits.dtype)
        
        # Add to logits
        # This biases the attention towards semantically similar patches
        return attn_logits + self.similarity_weight * sim_map
    
    def enhance_attention(self, attn_weights, num_heads=None):
        """
        Legacy method for enhancing probabilities (post-softmax).
        Retained for compatibility but enhance_attention_logits is preferred.
        """
        # ... implementation ...
        return None
        if self.cached_similarity_map is None:
            return None
        
        sim_map = self.cached_similarity_map.clone()  # [B, num_patches, num_patches]
        B_sim, num_patches, _ = sim_map.shape
        
        # SC-CLIP processing: mean-center, scale by 3×, clip negatives to -inf
        sim_map = (sim_map - torch.mean(sim_map)) * 3.0
        sim_map[sim_map < 0.0] = float('-inf')
        
        # Repeat for each head: [B, N_p, N_p] -> [B*num_heads, N_p, N_p]
        if num_heads is not None:
            sim_map = sim_map.repeat(num_heads, 1, 1)  # [B*num_heads, num_patches, num_patches]
        
        # Convert to same dtype as attention weights
        sim_map = sim_map.to(attn_weights.dtype)
        
        # Apply softmax to get similarity-based attention weights
        sim_attn = F.softmax(sim_map, dim=-1)  # [B*num_heads, num_patches, num_patches]
        
        # Determine if we need to pad for CLS token
        N_attn = attn_weights.shape[1]  # Total tokens in attention (may include CLS)
        if N_attn == num_patches:
            # No CLS token in attention, return as-is
            return sim_attn
        elif N_attn == num_patches + 1:
            # CLS token present at position 0, pad similarity attention
            BH = sim_attn.shape[0]
            device = sim_attn.device
            dtype = sim_attn.dtype
            
            # Create padded [BH, N, N] with zeros for CLS row/column
            sim_attn_padded = torch.zeros(BH, N_attn, N_attn, device=device, dtype=dtype)
            sim_attn_padded[:, 1:, 1:] = sim_attn  # Patch-patch similarities in bottom-right
            # CLS row and column remain zero — similarity doesn't affect CLS interactions
            
            return sim_attn_padded
        else:
            # Unexpected shape, return None to fall back to base attention
            return None
    
    def clear_cache(self):
        """Clear the cached similarity map."""
        self.cached_similarity_map = None
    
    def forward(self, final_features, mid_features):
        """
        Forward pass - compute and cache similarity map, return original features.
        The actual enhancement happens in custom_attn via enhance_attention().
        
        Args:
            final_features: [B, N, D] features (passed through unchanged)
            mid_features: [B, N, D] features from mid layer (for similarity computation)
            
        Returns:
            final_features: [B, N, D] unchanged features
        """
        # Cache similarity map for use in attention computation
        self.cache_similarity_map(mid_features)
        
        # Return features unchanged - enhancement happens in attention
        return final_features
    
    def get_similarity_map(self, mid_features):
        """
        Get the similarity map for visualization/analysis.
        
        Args:
            mid_features: [B, N, D] features from mid layer
            
        Returns:
            similarity_map: [B, N, N] pairwise cosine similarities
        """
        return self.compute_similarity_map(mid_features)
