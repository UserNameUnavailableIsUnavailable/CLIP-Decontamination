
import sys
import torch
import torch.nn as nn
from unittest.mock import MagicMock

# Mock dependencies
sys.modules['mmseg'] = MagicMock()
sys.modules['mmseg.models'] = MagicMock()
sys.modules['mmseg.models.segmentors'] = MagicMock()
sys.modules['mmseg.models.data_preprocessor'] = MagicMock()
sys.modules['mmengine'] = MagicMock()
sys.modules['mmengine.structures'] = MagicMock()
sys.modules['mmseg.registry'] = MagicMock()
sys.modules['open_clip'] = MagicMock()
sys.modules['BLIP'] = MagicMock()
sys.modules['BLIP.models'] = MagicMock()
sys.modules['BLIP.models.blip_retrieval'] = MagicMock()
sys.modules['gem'] = MagicMock()
sys.modules['simfeatup_dev'] = MagicMock()
sys.modules['simfeatup_dev.upsamplers'] = MagicMock()
sys.modules['prompts'] = MagicMock()
sys.modules['prompts.imagenet_template'] = MagicMock()

# Mock MODELS registry
mock_registry = MagicMock()
def register_module():
    def decorator(cls):
        return cls
    return decorator
mock_registry.register_module = register_module
sys.modules['mmseg.registry'].MODELS = mock_registry

# Import target
from segearth_segmentor import Segmentor

def test_segmentor_call():
    print("--- Testing Segmentor Integration ---")
    
    # Mocking
    segmentor = Segmentor(
        clip_type='CLIP',
        vit_type='ViT-B/16',
        model_type='SegEarth',
        name_path='configs/cls_chn6-cug.txt', # Needs to be a valid path
        device=torch.device('cpu'),
        apply_sim_feat_up=True,
        apply_similarity_enhancement=True,
        sim_feat_up_cfg={'model_name': 'jbu_one', 'model_path': 'dummy'}
    )
    
    # Mock net
    segmentor.net = MagicMock()
    # Mock encode_image to return features [B, N, D]
    B, N, D = 1, 197, 512 # 14x14 + 1
    segmentor.net.encode_image.return_value = torch.randn(B, N, D)
    segmentor.net.visual = MagicMock()
    segmentor.net.visual.patch_size = (16, 16)
    
    # Mock upsampler
    segmentor.upsampler = MagicMock()
    # If upsampling is applied, it returns high-res features
    # Let's say input image is 224x224. Patch size 16.
    # Original features: 14x14 = 196 (plus cls token handling)
    # Upsampled to image size: 224x224 = 50176
    H, W = 224, 224
    N_high = H * W
    def upsample_side_effect(x, img):
        # x is [1, D, h, w]
        # returns [1, D, H, W]
        return torch.randn(1, D, H, W)
    segmentor.upsampler.side_effect = upsample_side_effect

    # Create dummy query file
    with open('configs/cls_chn6-cug.txt', 'w') as f:
        f.write('cat\ndog\n')

    # Re-init to pick up file
    segmentor = Segmentor(
        clip_type='CLIP',
        vit_type='ViT-B/16', # Use B/16 to match patch size
        model_type='SegEarth',
        name_path='configs/cls_chn6-cug.txt',
        device=torch.device('cpu'),
        apply_sim_feat_up=True, # Enable upsampling
        apply_similarity_enhancement=True,
        sim_feat_up_cfg={'model_name': 'jbu_one', 'model_path': 'dummy'}
    )
    
    segmentor.net = MagicMock()
    segmentor.net.encode_image.return_value = torch.randn(B, 197, 512) # 1CLS + 196
    segmentor.net.visual = MagicMock()
    segmentor.net.visual.patch_size = (16, 16)
    segmentor.upsampler = MagicMock()
    segmentor.upsampler.side_effect = upsample_side_effect  
   
    # Mock sim enhancer to check if called
    segmentor.similarity_enhancer = MagicMock()
    def sim_side_effect(logits, features):
        print(f"SimilarityEnhancer CALLED with logits shape {logits.shape} and features shape {features.shape}")
        return logits
    segmentor.similarity_enhancer.side_effect = sim_side_effect

    # Forward feature
    img = torch.randn(1, 3, 224, 224)
    out = segmentor.forward_feature(img)
    
    # Verify calls
    if segmentor.similarity_enhancer.called:
        print("PASS: SimilarityEnhancer was called.")
    else:
        print("FAIL: SimilarityEnhancer was NOT called.")

if __name__ == "__main__":
    test_segmentor_call()
