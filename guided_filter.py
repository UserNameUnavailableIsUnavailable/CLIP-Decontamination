"""
Learnable Guided Filter for Segmentation Refinement

Implements a differentiable Guided Filter that uses high-resolution features 
(e.g., from SimFeatUp) to refine coarse segmentation logits.
Complexity: O(N) (Linear wrt image size), avoiding N^2 affinity matrices.

Reference: Deep Guided Filter (Wu et al., CVPR 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BoxFilter(nn.Module):
    def __init__(self, radius):
        super(BoxFilter, self).__init__()
        self.radius = radius

    def forward(self, x):
        # Box filter is essentially an average pooling
        # Padding ensures output size matches input size
        return F.avg_pool2d(x, kernel_size=2*self.radius+1, stride=1, padding=self.radius)

class LearnableFeatureGuidedFilter(nn.Module):
    """
    Refines coarse segmentation logits using high-resolution features (e.g. SimFeatUp) as guidance.
    
    Args:
        feature_dim (int): Dimensionality of the guidance features.
        logit_dim (int): Number of classes (dimensionality of logits).
        radius (int): Radius of the guided filter window.
        eps (float): Regularization parameter.
    """
    def __init__(self, feature_dim, logit_dim, radius=2, eps=1e-5):
        super(LearnableFeatureGuidedFilter, self).__init__()
        self.radius = radius
        self.eps = eps
        self.box_filter = BoxFilter(radius)
        
        # Learnable projection: adapt generic features to guidance map
        # We project features to match the number of classes (logit_dim)
        # so we can perform channel-wise guided filtering.
        # This allows Class A to use different boundary cues than Class B.
        self.guidance_proj = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=1),
            nn.InstanceNorm2d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 2, logit_dim, kernel_size=1) 
        )
        
        # Initialize to identity-like behavior to start
        # This prevents the filter from destroying logits initially
        # (Though we are training-free, a good init helps heuristic adaptation)
        nn.init.constant_(self.guidance_proj[-1].weight, 0)
        nn.init.constant_(self.guidance_proj[-1].bias, 0)


    def forward(self, logits, guidance_features):
        """
        Args:
            logits (Tensor): Coarse segmentation logits [B, C, H, W] (the 'p' input)
            guidance_features (Tensor): High-res features [B, F, H, W] (the 'I' guidance)
            
        Returns:
            refined_logits (Tensor): [B, C, H, W]
        """
        # 1. Project guidance features to obtain 'I'
        # I shape: [B, C, H, W]
        I = self.guidance_proj(guidance_features)
        
        # Normalize Guidance I to 0-1 range for stability (optional but good for GF)
        I = torch.sigmoid(I) 

        # Logits 'p'
        p = logits
        
        # Ensure dimensions match (upsample logits to match features if needed)
        if I.shape[-2:] != p.shape[-2:]:
            p = F.interpolate(p, size=I.shape[-2:], mode='bilinear', align_corners=False)

        # 2. Compute Mean parameters (The Guided Filter Algorithm)
        # All operations are element-wise or local (avg_pool), so O(N)
        
        mean_I = self.box_filter(I)
        mean_p = self.box_filter(p)
        mean_Ip = self.box_filter(I * p)
        mean_II = self.box_filter(I * I)

        # 3. Covariance and Variance
        cov_Ip = mean_Ip - mean_I * mean_p
        var_I = mean_II - mean_I * mean_I

        # 4. Linear Coefficients A and b
        # a = cov_Ip / (var_I + eps)
        # b = mean_p - a * mean_I
        a = cov_Ip / (var_I + self.eps)
        b = mean_p - a * mean_I

        # 5. Compute Mean of A and b
        mean_a = self.box_filter(a)
        mean_b = self.box_filter(b)

        # 6. Refine Output
        # q = mean_a * I + mean_b
        q = mean_a * I + mean_b

        return q
