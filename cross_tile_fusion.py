"""
Cross-Tile Semantic Fusion Module

Inspired by Swin Transformer's shifted window mechanism, this training-free module
fuses semantics between adjacent tiles during sliding window inference.

Key ideas:
1. Tiles processed sequentially lack cross-tile context (boundary artifacts).
2. Shifted window attention enables information flow between tiles.
3. Training-free fusion: use feature similarity and spatial proximity.

Mechanism:
- Cache features from overlapping regions of previous tiles
- Fuse current tile features with cached neighboring tile features
- Use attention-based or convolution-based fusion weighted by similarity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List


class CrossTileFusion(nn.Module):
    """
    Training-free cross-tile semantic fusion module.
    
    During sliding window inference, this module:
    1. Maintains a spatial cache of features from processed tiles
    2. Retrieves neighboring tile features for overlap regions
    3. Fuses current and cached features via attention or weighted averaging
    4. Updates cache with current tile features
    
    Args:
        fusion_mode: 'attention' (self-attention between tiles) or 'weighted' (similarity-weighted mean)
        overlap_ratio: Ratio of overlap between tiles (default: 0.5 for stride=112, crop=224)
        cache_boundary_width: Number of boundary patches to cache per tile (default: 2)
        fusion_strength: Maximum blending coefficient for fusion (0=no fusion, 1=full fusion)
        adaptive_fusion: If True, compute threshold and strength from similarity stats
        similarity_threshold: Optional fixed threshold (used only when adaptive_fusion=False)
    """
    
    def __init__(
        self,
        fusion_mode: str = 'weighted',
        overlap_ratio: float = 0.5,
        cache_boundary_width: int = 2,
        fusion_strength: float = 0.3,
        adaptive_fusion: bool = True,
        similarity_threshold: Optional[float] = None
    ):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.overlap_ratio = overlap_ratio
        self.cache_boundary_width = cache_boundary_width
        self.fusion_strength = fusion_strength
        self.adaptive_fusion = adaptive_fusion
        self.similarity_threshold = similarity_threshold
        
        # Spatial cache: stores features from processed tiles
        # Key: (h_idx, w_idx), Value: boundary features
        self.tile_cache: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        
        assert fusion_mode in ['attention', 'weighted'], \
            f"fusion_mode must be 'attention' or 'weighted', got {fusion_mode}"
    
    def reset_cache(self):
        """Clear tile cache (call at start of each image)."""
        self.tile_cache.clear()
    
    def extract_boundaries(
        self,
        features: torch.Tensor,
        patch_h: int,
        patch_w: int
    ) -> Dict[str, torch.Tensor]:
        """
        Extract boundary features from a tile.
        
        Args:
            features: [B, N, C] patch features
            patch_h, patch_w: spatial dimensions of patch grid
            
        Returns:
            Dict with keys 'top', 'bottom', 'left', 'right', each [B, boundary_len, C]
        """
        B, N, C = features.shape
        features_grid = features.view(B, patch_h, patch_w, C)  # [B, H, W, C]
        
        boundaries = {}
        bw = self.cache_boundary_width
        
        # Top: first bw rows
        boundaries['top'] = features_grid[:, :bw, :, :].reshape(B, -1, C)
        # Bottom: last bw rows
        boundaries['bottom'] = features_grid[:, -bw:, :, :].reshape(B, -1, C)
        # Left: first bw columns
        boundaries['left'] = features_grid[:, :, :bw, :].reshape(B, -1, C)
        # Right: last bw columns
        boundaries['right'] = features_grid[:, :, -bw:, :].reshape(B, -1, C)
        
        return boundaries
    
    def get_neighbor_features(
        self,
        h_idx: int,
        w_idx: int,
        direction: str
    ) -> Optional[torch.Tensor]:
        """
        Retrieve cached boundary features from neighboring tile.
        
        Args:
            h_idx, w_idx: current tile indices
            direction: 'top', 'bottom', 'left', 'right'
            
        Returns:
            Neighbor boundary features [B, boundary_len, C] or None
        """
        # Map direction to neighbor tile index
        neighbor_map = {
            'top': (h_idx - 1, w_idx),      # tile above
            'bottom': (h_idx + 1, w_idx),   # tile below
            'left': (h_idx, w_idx - 1),     # tile to left
            'right': (h_idx, w_idx + 1)     # tile to right
        }
        
        neighbor_idx = neighbor_map[direction]
        
        if neighbor_idx not in self.tile_cache:
            return None
        
        # Get opposite boundary from neighbor
        opposite = {
            'top': 'bottom',
            'bottom': 'top',
            'left': 'right',
            'right': 'left'
        }
        
        return self.tile_cache[neighbor_idx].get(opposite[direction])
    
    def fuse_with_attention(
        self,
        current: torch.Tensor,
        neighbor: torch.Tensor,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Fuse current and neighbor features using self-attention.
        
        Args:
            current: [B, N_curr, C]
            neighbor: [B, N_neigh, C]
            
        Returns:
            fused_current: [B, N_curr, C]
        """
        B, N_curr, C = current.shape
        N_neigh = neighbor.size(1)
        dtype = current.dtype
        
        # Concatenate for joint attention
        combined = torch.cat([current, neighbor], dim=1)  # [B, N_curr+N_neigh, C]
        
        # Compute self-attention weights (simplified, no learnable projection)
        # Q = current, K = combined
        q = current  # [B, N_curr, C]
        k = combined  # [B, N_curr+N_neigh, C]
        
        # Attention scores
        scale = torch.tensor(C ** 0.5, dtype=dtype, device=current.device)
        attn = torch.bmm(q, k.transpose(1, 2)) / scale  # [B, N_curr, N_curr+N_neigh]
        attn = F.softmax(attn, dim=-1)
        
        # Weighted sum
        fused = torch.bmm(attn, combined)  # [B, N_curr, C]
        
        # Blend with original
        fusion_strength = torch.tensor(self.fusion_strength, dtype=dtype, device=current.device)
        fused = current * (1 - fusion_strength) + fused * fusion_strength
        
        return fused
    
    def fuse_with_similarity(
        self,
        current: torch.Tensor,
        neighbor: torch.Tensor,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Fuse current and neighbor features using cosine similarity weighting.
        
        Args:
            current: [B, N_curr, C]
            neighbor: [B, N_neigh, C]
            
        Returns:
            fused_current: [B, N_curr, C]
        """
        B, N_curr, C = current.shape
        N_neigh = neighbor.size(1)
        dtype = current.dtype
        
        # Normalize
        current_norm = current / (current.norm(dim=-1, keepdim=True) + eps)
        neighbor_norm = neighbor / (neighbor.norm(dim=-1, keepdim=True) + eps)
        
        # Compute pairwise similarity: [B, N_curr, N_neigh]
        sim = torch.bmm(current_norm, neighbor_norm.transpose(1, 2))
        
        if self.adaptive_fusion or self.similarity_threshold is None:
            sim_mean = sim.mean(dim=-1, keepdim=True)
            sim_std = sim.std(dim=-1, keepdim=True)
            threshold = sim_mean + sim_std
            sim_margin = torch.relu(sim - threshold)
            sim_weight_raw = sim_margin.pow(2)
            sim_sum = sim_weight_raw.sum(dim=-1, keepdim=True) + eps
            weights = sim_weight_raw / sim_sum  # [B, N_curr, N_neigh]
            local_strength = sim_margin.mean(dim=-1, keepdim=True).clamp(0.0, 1.0)
        else:
            sim_mask = (sim > self.similarity_threshold).to(dtype=dtype)
            sim_masked = sim * sim_mask
            sim_sum = sim_masked.sum(dim=-1, keepdim=True) + eps
            weights = sim_masked / sim_sum  # [B, N_curr, N_neigh]
            local_strength = torch.ones_like(sim_sum, dtype=dtype)
        
        # Weighted average of neighbors
        neighbor_agg = torch.bmm(weights, neighbor)  # [B, N_curr, C]
        
        # Blend with current
        fusion_strength = torch.tensor(self.fusion_strength, dtype=dtype, device=current.device)
        fusion_strength = fusion_strength * local_strength
        fused = current * (1 - fusion_strength) + neighbor_agg * fusion_strength
        
        return fused
    
    def fuse_boundaries(
        self,
        features: torch.Tensor,
        boundaries: Dict[str, torch.Tensor],
        h_idx: int,
        w_idx: int,
        patch_h: int,
        patch_w: int
    ) -> torch.Tensor:
        """
        Fuse boundary regions with cached neighbor features.
        
        Args:
            features: [B, N, C] current tile features
            boundaries: extracted boundaries from current tile
            h_idx, w_idx: current tile position
            patch_h, patch_w: patch grid dimensions
            
        Returns:
            fused_features: [B, N, C] with boundaries fused
        """
        B, N, C = features.shape
        features_grid = features.view(B, patch_h, patch_w, C)
        bw = self.cache_boundary_width
        
        # Fuse each boundary direction
        for direction in ['top', 'bottom', 'left', 'right']:
            neighbor_feats = self.get_neighbor_features(h_idx, w_idx, direction)
            
            if neighbor_feats is None:
                continue  # No cached neighbor in this direction
            
            current_boundary = boundaries[direction]  # [B, boundary_len, C]
            
            # Apply fusion
            if self.fusion_mode == 'attention':
                fused_boundary = self.fuse_with_attention(current_boundary, neighbor_feats)
            else:  # 'weighted'
                fused_boundary = self.fuse_with_similarity(current_boundary, neighbor_feats)
            
            # Write back to grid
            if direction == 'top':
                features_grid[:, :bw, :, :] = fused_boundary.view(B, bw, patch_w, C)
            elif direction == 'bottom':
                features_grid[:, -bw:, :, :] = fused_boundary.view(B, bw, patch_w, C)
            elif direction == 'left':
                features_grid[:, :, :bw, :] = fused_boundary.view(B, patch_h, bw, C)
            elif direction == 'right':
                features_grid[:, :, -bw:, :] = fused_boundary.view(B, patch_h, bw, C)
        
        return features_grid.reshape(B, N, C)
    
    def forward(
        self,
        features: torch.Tensor,
        h_idx: int,
        w_idx: int,
        patch_h: int,
        patch_w: int
    ) -> torch.Tensor:
        """
        Apply cross-tile fusion to current tile features.
        
        Args:
            features: [B, N, C] patch features from current tile
            h_idx, w_idx: tile indices in sliding window grid
            patch_h, patch_w: spatial dimensions of patch grid
            
        Returns:
            fused_features: [B, N, C] with cross-tile fusion applied
        """
        # Extract boundaries
        boundaries = self.extract_boundaries(features, patch_h, patch_w)
        
        # Fuse with cached neighbors
        fused_features = self.fuse_boundaries(
            features, boundaries, h_idx, w_idx, patch_h, patch_w
        )
        
        # Update cache with current tile
        self.tile_cache[(h_idx, w_idx)] = boundaries
        
        return fused_features


def get_cross_tile_fusion(
    enabled: bool = True,
    fusion_mode: str = 'weighted',
    fusion_strength: float = 0.5,
    adaptive_fusion: bool = True,
    similarity_threshold: Optional[float] = None,
    **kwargs
) -> Optional[CrossTileFusion]:
    """
    Factory function to create cross-tile fusion module.
    
    Args:
        enabled: Whether to enable cross-tile fusion
        fusion_mode: 'attention' or 'weighted'
        fusion_strength: Blending strength
        **kwargs: Additional arguments for CrossTileFusion
        
    Returns:
        CrossTileFusion module or None if disabled
    """
    if not enabled:
        return None
    
    return CrossTileFusion(
        fusion_mode=fusion_mode,
        fusion_strength=fusion_strength,
        adaptive_fusion=adaptive_fusion,
        similarity_threshold=similarity_threshold,
        **kwargs
    )
