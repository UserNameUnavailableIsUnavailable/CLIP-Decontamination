"""
Similarity-based Feature Refinement Module (Post-processing)

Refines segmentation logits using self-similarity of visual features.
Aligned with SC-CLIP / SFP / MaskRefine approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimilarityRefiner(nn.Module):
    """
    Refines segmentation logits using the cosine similarity of the final visual features.
    
    L_refined = L + alpha * (S @ L)
    
    S is the column-normalized (or softmaxed) similarity matrix.
    This propagates high-confidence predictions to visually similar neighbors.
    """
    
    def __init__(self, alpha=0.4, temperature=0.15): 
        # Default alpha=0.4, temperature=0.15 found effective in literature (e.g. S-CLIP, MaskRefine)
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        print(f"Refining logits with alpha={self.alpha}, temp={self.temperature}")
        
    def forward(self, logits, visual_features):
        return self.refine_segmentation_logits(logits, visual_features)
        
    def refine_segmentation_logits(self, logits, visual_features):
        """
        Args:
            logits: [B, C, H, W] or [B, N, C] (Segmentation Scores)
            visual_features: [B, D, H, W] or [B, N, D] (Feature Map)
            
        Returns:
            refined_logits: Same shape as input logits.
        """
        # Detect input format
        is_spatial = (logits.ndim == 4)
        
        if is_spatial:
            B, C, H, W = logits.shape
            N = H * W
            # Flatten spatial dims
            logits_flat = logits.view(B, C, N).permute(0, 2, 1)  # [B, N, C]
            
            # visual_features likely matches H, W of logits?
            # If not, interpolate visual features to match logits size?
            # visual_features usually comes from backbone (small) while logits are upsampled (large).
            # We strictly need visual features at the same resolution as the similarity map desired.
            # Refining at high-res (upsampled logits) using upsampled features is slow (N^2).
            # Better to refine at low-res (backbone output) then upsample.
            
            if visual_features.ndim == 4:
                # Interpolate features to match logits resolution if needed? 
                # Or interpolate logits to feature resolution?
                # Usually we refine at feature resolution (small N).
                feature_H, feature_W = visual_features.shape[2], visual_features.shape[3]
                if (feature_H != H) or (feature_W != W):
                     # Resize features to match logits? Or resize logits to match features?
                     # Resizing features to high res (e.g. 512x512) -> N=262k -> Interaction matrix 68B elements -> OOM.
                     # We MUST refine at low resolution (feature map size).
                     pass 
                
                feats_flat = visual_features.view(B, -1, feature_H * feature_W).permute(0, 2, 1) # [B, N_feat, D]
            else:
                B_f, N_f, D_f = visual_features.shape
                feats_flat = visual_features

        else:
            # [B, N, C]
            logits_flat = logits 
            feats_flat = visual_features # [B, N, D]
            B, N, C = logits.shape

        # Verify shapes match for BMM
        if logits_flat.shape[1] != feats_flat.shape[1]:
            # This happens if logits are upsampled but features are not.
            # We should assume the user passes compatible tensors.
            # If not, we skip refinement to avoid crash.
            print(f"Warning: Shape mismatch in SimilarityRefiner: Logits {logits_flat.shape} vs Feats {feats_flat.shape}. Skipping.")
            return logits

        # Normalize features
        feats_norm = F.normalize(feats_flat, dim=-1) # [B, N, D]
        
        # Compute Affinity Matrix A = F @ F.T
        # [B, N, N]
        affinity = torch.bmm(feats_norm, feats_norm.transpose(1, 2))
        
        # Apply Softmax to get Transition Matrix S
        S = F.softmax(affinity / self.temperature, dim=-1) # [B, N, N]
        
        # Refine: L_new = L_old + alpha * (S @ L_old)
        smoothed_logits = torch.bmm(S, logits_flat)
        
        refined_logits_flat = logits_flat + self.alpha * smoothed_logits
        
        # Restore shape
        if is_spatial:
            refined_logits = refined_logits_flat.permute(0, 2, 1).view(B, C, H, W)
        else:
            refined_logits = refined_logits_flat
            
        return refined_logits
