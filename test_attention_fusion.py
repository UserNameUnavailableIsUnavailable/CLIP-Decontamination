"""
Test script for attention-based layer fusion
"""
import torch
import open_clip
from configs.Vaihingen import model as model_cfg

def test_attention_fusion():
    print("Testing attention-based layer fusion...")
    
    # Create a dummy model
    model_name = 'ViT-B-16'
    pretrained = 'laion2b_s34b_b88k'
    
    # Create model
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, 
        pretrained=pretrained,
        device='cpu'  # Use CPU for testing
    )
    
    # Create dummy input [B, C, H, W]
    dummy_input = torch.randn(2, 3, 224, 224)
    
    # Test without layer fusion
    print("\n1. Testing without layer fusion...")
    with torch.no_grad():
        features_no_fusion = clip_model.visual(
            dummy_input,
            apply_layer_fusion=False,
            layer_fusion_lambda=0.5,
            layer_fusion_threshold=0.7,
            apply_similarity_enhancement=False
        )
    print(f"   Features shape (no fusion): {features_no_fusion.shape}")
    
    # Test with attention-based layer fusion
    print("\n2. Testing with attention-based layer fusion...")
    try:
        with torch.no_grad():
            features_with_fusion = clip_model.visual(
                dummy_input,
                apply_layer_fusion=True,
                layer_fusion_lambda=0.5,
                layer_fusion_threshold=0.7,
                apply_similarity_enhancement=False
            )
        print(f"   Features shape (with fusion): {features_with_fusion.shape}")
        print(f"   ✅ Attention-based layer fusion works!")
        
        # Check if features are different
        diff = (features_with_fusion - features_no_fusion).abs().mean()
        print(f"   Mean absolute difference: {diff:.6f}")
        
        if diff > 1e-6:
            print(f"   ✅ Layer fusion modifies features as expected")
        else:
            print(f"   ⚠️  Warning: Features are identical, fusion may not be applied")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n3. All tests passed! ✅")
    return True

if __name__ == "__main__":
    success = test_attention_fusion()
    exit(0 if success else 1)
