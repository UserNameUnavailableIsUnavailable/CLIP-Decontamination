
import os
import sys
import torch
from mmengine.config import Config
from mmseg.registry import MODELS
from mmseg.models import build_segmentor

# Add current directory to sys.path
sys.path.append(os.getcwd())

# Import custom segmentors to ensure they are registered
import segmentor
import segearth_segmentor

def verify(enable_enhancement):
    print(f"\n===== Testing with apply_similarity_enhancement={enable_enhancement} =====")
    
    try:
        # Load base config
        cfg = Config.fromfile('configs/base_config.py')
        
        # Override parameter
        cfg.model['apply_similarity_enhancement'] = enable_enhancement
        
        # Provide dummy data for required arguments NOT in base config
        if 'name_path' not in cfg.model:
             cfg.model['name_path'] = 'configs/cls_potsdam.txt'
        
        # Ensure the file exists
        if not os.path.exists(cfg.model['name_path']):
             with open(cfg.model['name_path'], 'w') as f:
                 f.write("dummy_class")
        
        print(f"Building model with type: {cfg.model.type}")
        model = MODELS.build(cfg.model)
        print("Model built successfully.")

    except Exception as e:
        print(f"Failed to build model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify(True)
    verify(False)
