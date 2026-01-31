"""
Quick test script to verify outlier suppression module works correctly.
"""

import torch
from outlier_suppression import OutlierSuppressionModule, detect_outliers_by_attention


def test_outlier_detection():
    """Test outlier detection with synthetic attention weights."""
    print("=" * 60)
    print("Testing Outlier Detection")
    print("=" * 60)
    
    # Create synthetic attention weights [batch_size, num_heads, seq_len, seq_len]
    batch_size = 2
    num_heads = 12
    num_patches = 196  # 14x14 grid
    seq_len = num_patches + 1  # +1 for CLS token
    
    attn_weights = torch.rand(batch_size, num_heads, seq_len, seq_len)
    
    # Make some patches outliers (high CLS attention, low self-attention)
    for b in range(batch_size):
        # Patches 10, 50, 100 as outliers
        for patch_idx in [10, 50, 100]:
            token_idx = patch_idx + 1  # +1 for CLS
            # Low self-attention
            attn_weights[b, :, token_idx, token_idx] = 0.01
            # High CLS->patch attention
            attn_weights[b, :, 0, token_idx] = 0.5
    
    # Detect outliers
    outlier_indices = detect_outliers_by_attention(attn_weights, num_patches, top_k=10)
    
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Detected outliers shape: {outlier_indices.shape}")
    print(f"Outliers for batch 0: {outlier_indices[0].tolist()}")
    print(f"Expected to include: [10, 50, 100]")
    
    # Check if expected outliers were detected
    outliers_set = set(outlier_indices[0].tolist())
    if {10, 50, 100}.issubset(outliers_set):
        print("✓ Outlier detection working correctly!")
    else:
        print("✗ Some expected outliers were not detected")
    
    print()


def test_outlier_suppression():
    """Test full outlier suppression module."""
    print("=" * 60)
    print("Testing Outlier Suppression Module")
    print("=" * 60)
    
    # Create module
    suppressor = OutlierSuppressionModule(top_k=10)
    
    # Create synthetic feature map [B, C, H, W]
    B, C, H, W = 2, 768, 14, 14
    feature_map = torch.randn(B, C, H, W)
    
    # Create synthetic attention weights
    num_patches = H * W
    seq_len = num_patches + 1
    num_heads = 12
    attn_weights = torch.rand(B, num_heads, seq_len, seq_len)
    
    # Make some outliers
    for b in range(B):
        for patch_idx in [10, 50, 100]:
            token_idx = patch_idx + 1
            attn_weights[b, :, token_idx, token_idx] = 0.01
            attn_weights[b, :, 0, token_idx] = 0.5
    
    print(f"Input feature map shape: {feature_map.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # Apply suppression
    suppressed_features = suppressor(feature_map, attn_weights, H, W)
    
    print(f"Output feature map shape: {suppressed_features.shape}")
    print(f"✓ Suppression completed successfully!")
    
    # Verify output shape matches input
    assert suppressed_features.shape == feature_map.shape, "Output shape mismatch!"
    print("✓ Output shape matches input!")
    
    # Verify some features were modified
    diff = (feature_map - suppressed_features).abs().sum()
    if diff > 0:
        print(f"✓ Features were modified (total change: {diff.item():.4f})")
    else:
        print("✗ No features were modified")
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Outlier Suppression Module Test Suite")
    print("=" * 60 + "\n")
    
    test_outlier_detection()
    test_outlier_suppression()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
