
import sys
import os

# Add the current directory to sys.path so we can import the config
sys.path.append(os.getcwd())

try:
    from configs.base_config import model
    print(f"model['apply_similarity_enhancement'] = {model.get('apply_similarity_enhancement', 'NOT FOUND')}")
except Exception as e:
    print(f"Error loading config: {e}")
