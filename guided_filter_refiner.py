"""
Learnable Guided Filter Refiner Module

Replaces the O(N^2) Similarity Refiner with an O(N) Guided Filter approach.
Uses SimFeatUp features as the guidance map to refine segmentation logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GuidedFilterRefiner(nn.Module):
    def __init__(self, feature_dim, num_classes, radius=2, eps=1e-2):
        super().__init__()
        self.radius = radius
        self.eps = eps
        
        # Initialize projection if we want learnable (though inference might be frozen).
        # We'll use direct feature slicing for speed and robustness (see forward).
        # self.guide_conv = nn.Identity() 
    
    def box_filter(self, x, r):
        return F.avg_pool2d(x, kernel_size=2*r+1, stride=1, padding=r)

    def forward(self, logits, visual_features):
        """
        Refines logits using visual_features as guidance.
        Args:
            logits: [B, C, H, W]
            visual_features: [B, D, H, W]
        """
        if visual_features.ndim == 3:
             # If flattened [B, N, D], reshape
             B, N, D = visual_features.shape
             H, W = logits.shape[-2:]
             if N == H*W:
                 visual_features = visual_features.permute(0, 2, 1).view(B, D, H, W)
        
        # Ensure float32 for precision in PCA and filtering
        # Avoids errors if inputs are half/bfloat16
        if logits.dtype != torch.float32:
             logits = logits.float()
        if visual_features.dtype != torch.float32:
             visual_features = visual_features.float()

        B, C, H, W = logits.shape
        D = visual_features.shape[1]
        N_pixels = H * W
        
        # Use PCA to project features to 1 dimension for guidance (O(N) with D << N)
        # Features: [B, D, H, W] -> [B, D, N]
        
        # Flatten spatial dims
        x_flat = visual_features.view(B, D, N_pixels) # [B, D, N]
        
        # Center features
        x_mean = x_flat.mean(dim=2, keepdim=True)
        x_centered = x_flat - x_mean
        
        # Compute covariance matrix [B, D, D]
        # normalize by N-1
        cov = torch.bmm(x_centered, x_centered.transpose(1, 2)) / (N_pixels - 1)
        
        # Eigen decomposition
        # torch.linalg.eigh is for symmetric matrices (faster/stable)
        # It returns eigenvalues and eigenvectors in ascending order.
        eigenvalues, eigenvectors = torch.linalg.eigh(cov) 
        
        # Top 1 component direction: last column
        # eigenvectors: [B, D, D], take the last column vector
        top_component = eigenvectors[:, :, -1:] # [B, D, 1]
        
        # Project centered features onto the top component
        # I_flat = w^T * x
        projection_vector = top_component.transpose(1, 2) # [B, 1, D]
        I_flat = torch.bmm(projection_vector, x_centered) # [B, 1, N]
        
        # Reshape to image [B, 1, H, W]
        I = I_flat.view(B, 1, H, W)
             
        # Normalize guidance map to [0, 1] for stable 'eps'
        I_min = I.amin(dim=(2, 3), keepdim=True)
        I_max = I.amax(dim=(2, 3), keepdim=True)
        I = (I - I_min) / (I_max - I_min + 1e-5)
             
        p = logits
        r = self.radius
        eps = self.eps

        # Mean filters
        mean_I = self.box_filter(I, r)
        mean_p = self.box_filter(p, r)
        mean_Ip = self.box_filter(I * p, r)
        mean_II = self.box_filter(I * I, r)

        # Covariance of (I, p) and var(I)
        # I is [B, 1, H, W], p is [B, C, H, W]
        # Automatic broadcasting handles the channel dimension
        
        cov_Ip = mean_Ip - mean_I * mean_p # [B, C, H, W]
        var_I = mean_II - mean_I * mean_I  # [B, 1, H, W]

        # Linear coefficients A, b
        # var_I broadcasts to C channels
        a = cov_Ip / (var_I + eps) # [B, C, H, W]
        b = mean_p - a * mean_I    # [B, C, H, W]

        # Mean coefficients
        mean_a = self.box_filter(a, r)
        mean_b = self.box_filter(b, r)

        # Result
        q = mean_a * I + mean_b
        return q

