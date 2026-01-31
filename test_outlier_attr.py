"""Test if outlier suppressor attribute is properly set"""
import torch
from segmentor import SegmentorEx

# Create model with outlier suppression enabled
print("Creating model with apply_outlier_suppression=True...")
model = SegmentorEx(
    name_path='./configs/cls_vaihingen.txt',
    apply_outlier_suppression=True,
    vit_type='ViT-B/16',
    clip_type='CLIP',
    model_type='SegEarth',
    ignore_residual=True,
    cls_token_lambda=0.0,
)

# Check if the attribute is set
print(f"\nChecking attributes:")
print(f"  hasattr(model.net.visual, 'outlier_suppressor'): {hasattr(model.net.visual, 'outlier_suppressor')}")
if hasattr(model.net.visual, 'outlier_suppressor'):
    print(f"  model.net.visual.outlier_suppressor is None: {model.net.visual.outlier_suppressor is None}")
    print(f"  Type: {type(model.net.visual.outlier_suppressor)}")

print("\nDone!")
