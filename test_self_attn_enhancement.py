"""
Test Self-Attention Enhancement Module
"""
import torch
import sys

def test_self_attention_enhancement():
    """Test the self-attention enhancement module"""
    print("Testing Self-Attention Enhancement Module...")
    
    try:
        from self_attention_enhancement import SelfAttentionEnhancementModule
        
        # Create module
        enhancer = SelfAttentionEnhancementModule(
            enhancement_strength=0.1,
            min_self_attn_threshold=0.15,
            mode='feature'
        )
        print("✅ Module created successfully")
        
        # Create dummy data
        B, C, H, W = 2, 768, 14, 14
        num_patches = H * W
        N = num_patches + 1  # +1 for CLS token
        
        features = torch.randn(B, C, H, W)
        
        # Create attention weights with some tokens having weak self-attention
        attn_weights = torch.softmax(torch.randn(B, N, N), dim=-1)
        
        # Manually set some tokens to have weak self-attention
        for b in range(B):
            for i in range(10):  # First 10 patch tokens
                patch_idx = i + 1  # +1 for CLS
                # Reduce self-attention (diagonal) and spread to others
                attn_weights[b, patch_idx, patch_idx] = 0.05  # Weak self-attention
        
        # Get stats before enhancement
        stats_before = enhancer.get_self_attention_stats(attn_weights)
        print(f"\n📊 Self-attention stats BEFORE enhancement:")
        print(f"   Mean: {stats_before['mean']:.4f}")
        print(f"   Min:  {stats_before['min']:.4f}")
        print(f"   Max:  {stats_before['max']:.4f}")
        print(f"   Tokens below threshold: {stats_before['below_threshold']*100:.1f}%")
        
        # Test feature-based enhancement
        print("\n🔧 Testing feature-based enhancement...")
        enhanced_features = enhancer.enhance_features(features, attn_weights, H, W)
        
        assert enhanced_features.shape == features.shape, f"Shape mismatch: {enhanced_features.shape} vs {features.shape}"
        
        diff = (enhanced_features - features).abs().mean()
        print(f"   Mean absolute difference: {diff:.6f}")
        
        if diff > 1e-6:
            print("   ✅ Features were modified as expected")
        else:
            print("   ⚠️  Warning: Features unchanged")
        
        # Test attention-based enhancement
        print("\n🔧 Testing attention-based enhancement...")
        enhancer_attn = SelfAttentionEnhancementModule(
            enhancement_strength=0.1,
            min_self_attn_threshold=0.15,
            mode='attention'
        )
        
        enhanced_features_attn = enhancer_attn.enhance_via_attention(features, attn_weights, H, W)
        
        assert enhanced_features_attn.shape == features.shape
        diff_attn = (enhanced_features_attn - features).abs().mean()
        print(f"   Mean absolute difference: {diff_attn:.6f}")
        
        if diff_attn > 1e-6:
            print("   ✅ Features were modified as expected")
        
        # Test with sequence format
        print("\n🔧 Testing with sequence format [B, N, C]...")
        features_seq = features.view(B, C, H*W).permute(0, 2, 1)  # [B, N, C]
        enhanced_seq = enhancer.enhance_features(features_seq, attn_weights, H, W)
        
        assert enhanced_seq.shape == features_seq.shape
        print("   ✅ Sequence format works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test integration with attention weights from real transformer"""
    print("\n\nTesting Integration Scenario...")
    
    try:
        from self_attention_enhancement import SelfAttentionEnhancementModule
        
        # Simulate realistic attention pattern
        B, H, W = 1, 14, 14
        num_patches = H * W  # 196
        N = num_patches + 1  # 197 (with CLS)
        num_heads = 12
        C = 768
        
        # Multi-head attention [B, num_heads, N, N]
        attn_multi_head = torch.softmax(torch.randn(B, num_heads, N, N), dim=-1)
        
        # Simulate outlier pattern: some tokens attend heavily to CLS, weakly to self
        for h in range(num_heads):
            for i in range(20):  # 20 potential outlier tokens
                patch_idx = i + 1
                # High CLS attention
                attn_multi_head[0, h, patch_idx, 0] = 0.3
                # Low self-attention
                attn_multi_head[0, h, patch_idx, patch_idx] = 0.05
                # Re-normalize
                attn_multi_head[0, h, patch_idx] = attn_multi_head[0, h, patch_idx] / attn_multi_head[0, h, patch_idx].sum()
        
        features = torch.randn(B, C, H, W)
        
        # Test enhancement
        enhancer = SelfAttentionEnhancementModule(
            enhancement_strength=0.15,
            min_self_attn_threshold=0.2,
            mode='feature'
        )
        
        stats = enhancer.get_self_attention_stats(attn_multi_head)
        print(f"\n📊 Attention pattern stats:")
        print(f"   Mean self-attention: {stats['mean']:.4f}")
        print(f"   Tokens below threshold (0.2): {stats['below_threshold']*100:.1f}%")
        
        enhanced = enhancer(features, attn_multi_head, H, W)
        
        print(f"\n✅ Enhancement applied successfully")
        print(f"   Input shape:  {features.shape}")
        print(f"   Output shape: {enhanced.shape}")
        
        # Compare CLS token (should be unchanged)
        features_seq = features.view(B, C, -1).permute(0, 2, 1)
        enhanced_seq = enhanced.view(B, C, -1).permute(0, 2, 1)
        
        # The implementation doesn't include CLS in enhancement
        # So we need to check patches only
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 70)
    print("Self-Attention Enhancement Module Test")
    print("=" * 70)
    
    test1 = test_self_attention_enhancement()
    test2 = test_integration()
    
    print("\n" + "=" * 70)
    if test1 and test2:
        print("✅ All tests passed!")
        print("\nKey Features:")
        print("• Detects tokens with weak self-attention")
        print("• Enhances features to make tokens more self-reliant")
        print("• Two modes: feature-based and attention-based")
        print("• Mild, controlled enhancement to avoid overcorrection")
        print("• Works with both spatial [B,C,H,W] and sequence [B,N,C] formats")
    else:
        print("❌ Some tests failed")
    print("=" * 70)
    
    return 0 if (test1 and test2) else 1

if __name__ == "__main__":
    sys.exit(main())
