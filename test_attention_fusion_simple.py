"""
Simple syntax and import test for attention-based layer fusion
"""
import sys
import torch

def test_imports():
    """Test that all modules import correctly"""
    print("Testing imports...")
    try:
        import open_clip
        from open_clip.transformer import VisionTransformer
        from outlier_suppression import detect_outliers_by_attention, OutlierSuppressionModule
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_outlier_detection():
    """Test outlier detection function"""
    print("\nTesting outlier detection...")
    try:
        from outlier_suppression import detect_outliers_by_attention
        
        # Create dummy attention weights [batch_size, seq_len, seq_len]
        # seq_len = 1 (CLS) + 196 (patches for 14x14 grid)
        batch_size = 2
        seq_len = 197
        num_patches = 196
        
        attn_weights = torch.randn(batch_size, seq_len, seq_len).softmax(dim=-1)
        
        # Detect outliers
        outlier_indices = detect_outliers_by_attention(attn_weights, num_patches=num_patches, top_k=10)
        
        print(f"   Attention shape: {attn_weights.shape}")
        print(f"   Outlier indices shape: {outlier_indices.shape}")
        print(f"   Sample outlier indices (batch 0): {outlier_indices[0][:5]}")
        
        assert outlier_indices.shape == (batch_size, 10), f"Expected shape (2, 10), got {outlier_indices.shape}"
        print("✅ Outlier detection works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Outlier detection error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_attention_fusion_logic():
    """Test the attention fusion mathematical operations"""
    print("\nTesting attention fusion logic...")
    try:
        # Simulate attention fusion
        batch_size = 2
        num_heads = 12
        seq_len = 197
        
        # Create dummy attention maps
        attn_l = torch.randn(batch_size * num_heads, seq_len, seq_len).softmax(dim=-1)
        attn_l1 = torch.randn(batch_size * num_heads, seq_len, seq_len).softmax(dim=-1)
        
        # Fusion: A_{l+1} = λ*A_l + (1-λ)*A_{l+1}
        lambda_val = 0.5
        attn_fused = lambda_val * attn_l + (1 - lambda_val) * attn_l1
        
        print(f"   Attention L shape: {attn_l.shape}")
        print(f"   Attention L+1 shape: {attn_l1.shape}")
        print(f"   Fused attention shape: {attn_fused.shape}")
        
        # Average over heads
        attn_fused_avg = attn_fused.view(batch_size, num_heads, seq_len, seq_len).mean(dim=1)
        print(f"   Fused attention (avg heads) shape: {attn_fused_avg.shape}")
        
        # Test masking
        from outlier_suppression import detect_outliers_by_attention
        num_patches = seq_len - 1
        outlier_indices = detect_outliers_by_attention(attn_fused_avg, num_patches=num_patches, top_k=10)
        
        # Create mask
        attn_mask = torch.ones(batch_size, seq_len)
        for b in range(batch_size):
            outlier_positions = outlier_indices[b] + 1  # +1 for CLS
            attn_mask[b, outlier_positions] = 0.0
        
        # Apply mask
        attn_masked = attn_fused_avg * attn_mask.unsqueeze(1)
        
        # Normalize
        attn_normalized = attn_masked / (attn_masked.sum(dim=-1, keepdim=True) + 1e-8)
        
        print(f"   Masked attention shape: {attn_masked.shape}")
        print(f"   Normalized attention shape: {attn_normalized.shape}")
        print(f"   Sum of normalized attention (row 0): {attn_normalized[0, 0].sum():.4f}")
        
        # Test attention-weighted feature aggregation
        features = torch.randn(batch_size, seq_len, 768)
        features_weighted = torch.bmm(attn_normalized, features)
        
        print(f"   Features shape: {features.shape}")
        print(f"   Weighted features shape: {features_weighted.shape}")
        
        print("✅ Attention fusion logic works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Attention fusion logic error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("Testing Attention-Based Layer Fusion Implementation")
    print("=" * 60)
    
    all_passed = True
    all_passed &= test_imports()
    all_passed &= test_outlier_detection()
    all_passed &= test_attention_fusion_logic()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed!")
        print("=" * 60)
        return 0
    else:
        print("❌ Some tests failed")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
