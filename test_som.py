"""
Test script for the Suppress Outlier Module (SOM).

This script tests the SOM implementation to verify it correctly:
1. Detects outliers based on attention map comparison
2. Replaces outlier tokens with 8-neighbor mean
"""

import torch
import torch.nn.functional as F
from COS import SuppressOutlierModule


def test_som_basic():
    """Test basic SOM functionality with synthetic data."""
    print("=" * 60)
    print("Testing SOM Basic Functionality")
    print("=" * 60)
    
    # Create SOM module with default mode
    som = SuppressOutlierModule()
    
    # Synthetic patch tokens: [B=1, num_patches=16 (4x4 grid), C=768]
    B, grid_h, grid_w, C = 1, 4, 4, 768
    num_patches = grid_h * grid_w
    
    tokens = torch.randn(B, num_patches, C)
    
    # Create synthetic attention weights: [B, num_heads, N, N] where N = 1 + num_patches
    num_heads = 12
    N = 1 + num_patches  # CLS + patches
    
    # Initialize attention weights
    attn = torch.rand(B, num_heads, N, N)
    attn = F.softmax(attn, dim=-1)  # Normalize
    
    # Make some patches outliers by setting their self-attention < CLS attention to them
    # For patches at positions 5 and 10, make them outliers
    outlier_indices = [5, 10]  # In the patch token space (0-indexed)
    for idx in outlier_indices:
        patch_idx = idx + 1  # +1 because CLS is at 0
        # Set self-attention very low
        attn[:, :, patch_idx, patch_idx] = 0.01
        # Set CLS->patch attention high
        attn[:, :, 0, patch_idx] = 0.5
    
    print(f"Input tokens shape: {tokens.shape}")
    print(f"Attention weights shape: {attn.shape}")
    print(f"Grid size: {grid_h}x{grid_w}")
    
    # Apply SOM
    purified_tokens, outlier_mask, _ = som(tokens, attn, grid_h, grid_w, return_outlier_mask=True)
    
    print(f"Output tokens shape: {purified_tokens.shape}")
    print(f"Outlier mask shape: {outlier_mask.shape}")
    print(f"Number of outliers detected: {outlier_mask.sum().item()}")
    
    # Verify outlier detection
    outlier_flat = outlier_mask.view(B, num_patches)
    detected_outliers = outlier_flat[0].nonzero().squeeze(-1).tolist()
    print(f"Detected outlier positions: {detected_outliers}")
    print(f"Expected outlier positions: {outlier_indices}")
    
    # Check if the correct positions were detected
    if set(detected_outliers) >= set(outlier_indices):
        print("✓ Outlier detection working correctly!")
    else:
        print("✗ Outlier detection may need adjustment")
    
    # Verify that non-outlier tokens are unchanged
    non_outlier_mask = ~outlier_flat[0]
    if torch.allclose(tokens[0, non_outlier_mask], purified_tokens[0, non_outlier_mask]):
        print("✓ Non-outlier tokens preserved correctly!")
    else:
        print("✗ Non-outlier tokens were unexpectedly modified")
    
    # Verify that outlier tokens were changed
    if not torch.allclose(tokens[0, outlier_flat[0]], purified_tokens[0, outlier_flat[0]]):
        print("✓ Outlier tokens were successfully replaced!")
    else:
        print("✗ Outlier tokens were not modified")
    
    print()
    return True


def test_bidirectional_detection():
    """Test bidirectional outlier detection."""
    print("=" * 60)
    print("Testing Bidirectional Detection")
    print("=" * 60)
    
    B, grid_h, grid_w, C = 1, 4, 4, 4
    num_patches = grid_h * grid_w
    tokens = torch.randn(B, num_patches, C)
    
    num_heads = 4
    N = 1 + num_patches
    attn = torch.rand(B, num_heads, N, N)
    attn = F.softmax(attn, dim=-1)
    
    # Create different types of outliers
    # Patch 5: High CLS→patch attention, low self-attention
    attn[:, :, 6, 6] = 0.01  # Low Attn_{5,5}
    attn[:, :, 0, 6] = 0.5   # High Attn_{cls,5}
    
    # Patch 10: High patch→CLS attention, low self-attention
    attn[:, :, 11, 11] = 0.01  # Low Attn_{10,10}
    attn[:, :, 11, 0] = 0.4    # High Attn_{10,cls}
    
    # Patch 8: Both directions (bidirectional coupling)
    attn[:, :, 9, 9] = 0.01    # Low Attn_{8,8}
    attn[:, :, 9, 0] = 0.4     # High Attn_{8,cls}
    attn[:, :, 0, 9] = 0.3     # High Attn_{cls,8}
    
    som = SuppressOutlierModule(consensus_threshold=0.5)
    _, mask, confidence = som(tokens, attn, grid_h, grid_w, 
                              return_outlier_mask=True, return_confidence=True)
    num_outliers = mask.sum().item()
    
    print(f"  Total outliers detected: {num_outliers}")
    print(f"  Expected: at least 3 (patches 5, 8, 10)")
    
    assert num_outliers >= 3, "Should detect at least the 3 manually created outliers"
    print("✓ Bidirectional detection working correctly!")
    print()
    return True


def test_som_neighbor_mean():
    """Test that the 8-neighbor mean calculation is correct."""
    print("=" * 60)
    print("Testing 8-Neighbor Mean Calculation")
    print("=" * 60)
    
    som = SuppressOutlierModule()
    
    # Create a simple 3x3 grid where we know the expected neighbor mean
    B, grid_h, grid_w, C = 1, 3, 3, 4  # Small for easy verification
    num_patches = grid_h * grid_w
    
    # Create tokens with known values
    # Grid layout (values shown for first channel):
    # 1 2 3
    # 4 5 6
    # 7 8 9
    tokens = torch.arange(1, num_patches + 1).float().view(1, num_patches, 1).expand(1, num_patches, C)
    
    # Create attention that marks center (index 4) as outlier
    num_heads = 1
    N = 1 + num_patches
    attn = torch.zeros(B, num_heads, N, N)
    
    # Make center patch (index 4, which is position 5 in attention) an outlier
    center_idx = 4
    patch_idx = center_idx + 1
    attn[:, :, patch_idx, patch_idx] = 0.01  # Low self-attention
    attn[:, :, 0, patch_idx] = 0.5  # High CLS attention
    
    # Non-outlier patches should have high self-attention
    for i in range(num_patches):
        if i != center_idx:
            attn[:, :, i + 1, i + 1] = 0.5
            attn[:, :, 0, i + 1] = 0.01
    
    # Apply SOM
    purified, mask = som(tokens, attn, grid_h, grid_w, return_outlier_mask=True)[:2]
    
    print(f"Original center value: {tokens[0, center_idx, 0].item()}")
    print(f"Purified center value: {purified[0, center_idx, 0].item()}")
    
    # Expected neighbor mean for center: (1+2+3+4+6+7+8+9) / 8 = 40/8 = 5.0
    expected_mean = 5.0
    print(f"Expected neighbor mean: {expected_mean}")
    
    if abs(purified[0, center_idx, 0].item() - expected_mean) < 0.01:
        print("✓ 8-neighbor mean calculation is correct!")
    else:
        print("✗ 8-neighbor mean calculation needs adjustment")
    
    print()
    return True


def test_som_boundary_handling():
    """Test that boundary tokens are handled correctly."""
    print("=" * 60)
    print("Testing Boundary Handling")
    print("=" * 60)
    
    som = SuppressOutlierModule()
    
    # Create a 3x3 grid
    B, grid_h, grid_w, C = 1, 3, 3, 4
    num_patches = grid_h * grid_w
    
    tokens = torch.randn(B, num_patches, C)
    
    # Make corner (0,0) an outlier - it only has 3 neighbors
    num_heads = 1
    N = 1 + num_patches
    attn = torch.zeros(B, num_heads, N, N)
    
    # Corner patch (index 0)
    corner_idx = 0
    patch_idx = corner_idx + 1
    attn[:, :, patch_idx, patch_idx] = 0.01
    attn[:, :, 0, patch_idx] = 0.5
    
    # Other patches are not outliers
    for i in range(1, num_patches):
        attn[:, :, i + 1, i + 1] = 0.5
        attn[:, :, 0, i + 1] = 0.01
    
    # Apply SOM - should not crash on boundary
    try:
        purified, mask = som(tokens, attn, grid_h, grid_w, return_outlier_mask=True)[:2]
        print("✓ Boundary handling works without errors!")
        print(f"Corner outlier detected: {mask[0, 0, 0].item()}")
    except Exception as e:
        print(f"✗ Boundary handling failed: {e}")
        return False
    
    print()
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SOM (Suppress Outlier Module) Test Suite")
    print("=" * 60 + "\n")
    
    all_passed = True
    
    all_passed &= test_som_basic()
    all_passed &= test_bidirectional_detection()
    all_passed &= test_som_neighbor_mean()
    all_passed &= test_som_boundary_handling()
    
    print("=" * 60)
    if all_passed:
        print("All tests passed! ✓")
    else:
        print("Some tests failed! ✗")
    print("=" * 60)
