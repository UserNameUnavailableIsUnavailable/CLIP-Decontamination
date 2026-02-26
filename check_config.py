
import sys
from mmengine.config import Config

def check_config(config_path):
    print(f"Loading config: {config_path}")
    try:
        cfg = Config.fromfile(config_path)
        if 'model' in cfg:
            model_cfg = cfg.model
            apply_sim = model_cfg.get('apply_similarity_enhancement', 'NOT FOUND')
            print(f"model.apply_similarity_enhancement: {apply_sim}")
            
            sim_cfg = model_cfg.get('similarity_enhancement_cfg', 'NOT FOUND')
            print(f"model.similarity_enhancement_cfg: {sim_cfg}")
        else:
            print("Config has no 'model' key.")
            
    except Exception as e:
        print(f"Failed to load config: {e}")

if __name__ == "__main__":
    # Check base config first
    check_config('configs/base_config.py')
    # Check a derived config
    check_config('configs/cfg_chn6-cug.py')
