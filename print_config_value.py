import sys
import os
import argparse
from mmengine.config import Config
from mmseg.registry import MODELS
import segmentor
import segearth_segmentor
import inspect

def main():
    config_path = 'configs/cfg_potsdam.py'
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return

    print(f"Loading config from: {config_path}")
    try:
        cfg = Config.fromfile(config_path)
    except Exception as e:
        print(f"Error loading config: {e}")
        return
    
    # 1. Check apply_similarity_enhancement
    val = cfg.model.get('apply_similarity_enhancement', 'NOT_SET')
    print(f"\n[Config Check]")
    print(f"model.apply_similarity_enhancement = {val}")
    
    # 2. Check model type
    model_type = cfg.model.get('type', 'NOT_SET')
    print(f"model.type = '{model_type}'")
    
    # 3. Resolve Class
    model_class = MODELS.get(model_type)
    if model_class:
        print(f"Resolved Class: {model_class.__name__}")
        print(f"Defined in: {inspect.getfile(model_class)}")
    else:
        print(f"Could not resolve class for type '{model_type}'")

    # 4. Check imports of Refiners
    print(f"\n[Import Check]")
    try:
        from similarity_refiner import SimilarityRefiner
        print(f"Successfully imported SimilarityRefiner from similarity_refiner")
    except ImportError as e:
        print(f"Failed to import SimilarityRefiner: {e}")
        # Try finding file
        if os.path.exists('similarity_refiner.py'):
             print("File similarity_refiner.py exists.")
        else:
             print("File similarity_refiner.py MISSING.")

    try:
        from similarity_enhancement import SimilarityEnhancementModule
        print(f"Successfully imported SimilarityEnhancementModule from similarity_enhancement")
    except ImportError as e:
        print(f"Failed to import SimilarityEnhancementModule: {e}")

if __name__ == "__main__":
    main()
