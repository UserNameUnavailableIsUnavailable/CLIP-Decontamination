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
    
    def enhance_attention(self, attn_weights, num_heads=None):
        """
        Enhance attention weights by adding cached similarity map.
        
        Args:
            attn_weights: [B*num_heads, N, N] attention weights (before softmax)
                         where N = 1 + num_patches (includes CLS token)
            num_heads: Number of attention heads
            
        Returns:
            enhanced_attn: [B*num_heads, N, N] enhanced attention weights
        """
        if self.cached_similarity_map is None:
            return attn_weights
        
        sim_map = self.cached_similarity_map  # [B, num_patches, num_patches]
        B_sim, num_patches, _ = sim_map.shape
        
        # The attention includes CLS token, so N = 1 + num_patches
        # We need to pad similarity map with zeros for CLS token
        # Create [B, N, N] where N = 1 + num_patches
        N = num_patches + 1
        device = attn_weights.device
        dtype = attn_weights.dtype
            # Repeat for each head: [B, N, N] -> [B*num_heads, N, N]
        # Create padded similarity map with zeros for CLS row/column
        sim_map_padded = torch.zeros(B_sim, N, N, device=device, dtype=sim_map.dtype)
        sim_map_padded[:, 1:, 1:] = sim_map  # Put patch similarities in bottom-right
        
        # Expand similarity map to match attention shape [B*num_heads, N, N]
        if num_heads is not None:
            # Repeat for each head: [B, N, N] -> [B*num_heads, N, N]
            sim_map_padded = sim_map_padded.unsqueeze(1).expand(-1, num_heads, -1, -1)
            sim_map_padded = sim_map_padded.reshape(B_sim * num_heads, N, N)
        
        # Convert to same dtype as attention weights
        sim_map_padded = sim_map_padded.to(dtype)
        
        # Add weighted similarity map directly to attention weights (no softmax)
        # Softmax causes significant decay because:
        # 1. Softmax normalizes to sum=1, which drastically reduces the magnitude
        # 2. With N patches (~196), each softmax entry becomes ~1/N = 0.005
        # 3. This tiny contribution barely affects the original attention weights
        # Instead, we add the raw cosine similarity (range [-1, 1]) directly
        enhanced_attn = attn_weights + self.similarity_weight * sim_map_padded
        
        return enhanced_attn
    
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
