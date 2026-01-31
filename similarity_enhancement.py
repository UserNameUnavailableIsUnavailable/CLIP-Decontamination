"""
Similarity-based Feature Enhancement Module

Enhances features by adding self-similarity information from mid-layer representations.
Mid-layer features balance spatial and semantic information, making them ideal
for computing meaningful patch-to-patch similarities.

This is a completely training-free module with no learnable parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimilarityEnhancementModule(nn.Module):
    """
    Enhances features by adding self-similarity map computed from mid-layer features.
    
    The similarity map M[i,j] = CosSim(x_i, x_j) captures the likelihood that patches
    i and j belong to the same category. This similarity information is added directly
    to compensate for feature contamination.
    
    Args:
        temperature: Temperature for softmax over similarities (default: 1.0)
        add_self_similarity: Whether to include self-similarity in the map (default: False)
    """
    
    def __init__(self, temperature=1.0, add_self_similarity=False):
        super().__init__()
        self.temperature = temperature
        self.add_self_similarity = add_self_similarity
    
    def compute_similarity_map(self, features):
        """
        Compute pairwise cosine similarity map.
        
        Args:
            features: [B, N, D] patch features
            
        Returns:
            similarity_map: [B, N, N] pairwise cosine similarities
        """
        # Normalize features for cosine similarity
        features_norm = F.normalize(features, p=2, dim=-1)  # [B, N, D]
        
        # Compute similarity matrix: M[i,j] = CosSim(x_i, x_j)
        similarity_map = torch.bmm(features_norm, features_norm.transpose(1, 2))  # [B, N, N]
        
        # Optionally remove self-similarity
        if not self.add_self_similarity:
            B, N = similarity_map.shape[0], similarity_map.shape[1]
            mask = torch.eye(N, device=similarity_map.device).unsqueeze(0).expand(B, -1, -1)
            similarity_map = similarity_map * (1 - mask)
        
        return similarity_map
    
    def forward(self, final_features, mid_features):
        """
        Forward pass - calibrates features using self-similarity from mid-layer.
        
        In self-calibrated CLIP, similarity map is used to:
        1. Identify high-consensus (confident) regions via similarity scores
        2. Suppress low-consensus (uncertain) regions
        3. Refine features by aggregating from high-similarity neighbors
        
        Args:
            final_features: [B, N, D] features to enhance (from final layers)
            mid_features: [B, N, D] features from mid layer (for similarity computation)
            
        Returns:
            enhanced_features: [B, N, D] calibrated features
        """
        B, N, D = final_features.shape
        dtype = final_features.dtype  # Preserve input dtype (e.g., float16)
        
        # Compute similarity map from mid-layer features (in float32 for numerical stability)
        similarity_map = self.compute_similarity_map(mid_features.float())  # [B, N, N]
        
        # Compute confidence score for each patch based on maximum similarity to others
        # High max similarity = patch has confident neighbors with similar semantics
        # Shape: [B, N]
        confidence_scores, _ = similarity_map.max(dim=-1)
        
        # Apply temperature scaling to similarity map for refinement
        similarity_map_scaled = similarity_map / self.temperature
        similarity_weights = F.softmax(similarity_map_scaled, dim=-1)  # [B, N, N]
        
        # Aggregate features from similar neighbors (in float32 for bmm stability)
        # Shape: [B, N, N] @ [B, N, D] -> [B, N, D]
        refined_features = torch.bmm(similarity_weights, final_features.float())
        
        # Self-calibration: Use confidence to modulate between original and refined
        # Low confidence -> keep original (uncertain, don't change)
        # High confidence -> use refined (confident consensus exists)
        confidence_weights = confidence_scores.unsqueeze(-1)  # [B, N, 1]
        
        # Calibrated output: blend based on confidence (convert everything back to original dtype)
        calibrated_features = (
            confidence_weights.to(dtype) * refined_features.to(dtype) + 
            (1 - confidence_weights.to(dtype)) * final_features
        )
        
        return calibrated_features
    
    def get_similarity_map(self, mid_features):
        """
        Get the similarity map for visualization/analysis.
        
        Args:
            mid_features: [B, N, D] features from mid layer
            
        Returns:
            similarity_map: [B, N, N] pairwise cosine similarities
        """
        return self.compute_similarity_map(mid_features)
