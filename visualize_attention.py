
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms

img_path = "demo/img.png"

# Adjust path to include project root
import sys
sys.path.append(os.getcwd())

from segmentor import SegmentorEx

def get_attention_maps(model, img, device):
    """
    Run inference and return attention maps.
    """
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Handle fp16 if model is initialized in fp16
    if hasattr(model.net.visual, 'conv1') and model.net.visual.conv1.weight.dtype == torch.float16:
        img_tensor = img_tensor.half()
        
    print(f"Running inference on {device}...")
    
    # Run encode_image to get attentions
    with torch.no_grad():
        outputs = model.net.encode_image(
            img_tensor, 
            model_type='ClearCLIP', 
            ignore_residual=True, 
            output_cls_token=True,
            return_attentions=True
        )
    return outputs

def visualize_layers(model, img, device):
    outputs = get_attention_maps(model, img, device)
    cls_token_out, spatial_tokens, attentions = outputs
    
    # Grid dimensions
    h, w = 14, 14 
    patch_size = 16
    
    # Interactive selection
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title("Click to select a patch center")
    print("Please click on the image to select a point...")
    pts = plt.ginput(1, timeout=-1)
    plt.close()
    
    if not pts:
        print("No point selected. Using default center.")
        center_h, center_w = h // 2, w // 2
    else:
        x, y = pts[0]
        W_orig, H_orig = img.size
        # Map to [0, 14)
        grid_x = int((x / W_orig) * w)
        grid_y = int((y / H_orig) * h)
        center_w = max(0, min(w-1, grid_x))
        center_h = max(0, min(h-1, grid_y))
        print(f"Selected point: ({x:.1f}, {y:.1f}) -> Grid patch: ({center_h}, {center_w})")

    center_idx = center_h * w + center_w + 1 # +1 for CLS token offset
    
    # Process attentions
    # attentions is a list of [B, H, N, N] tensors
    # We want layers [0, 5, 7, 10]
    layer_idxs = [0, 5, 7, 10]
    num_layers = len(layer_idxs)
    
    # Visualization setup
    # Create two figures: one for attention grid, one for segmentation
    
    # 1. Attention Grid
    fig_attn = plt.figure(figsize=(24, 20), dpi=100)
    img_vis = img.resize((224, 224))
    
    for i, layer_idx in enumerate(layer_idxs):
        # ... processing existing logic ...
        # Get attention for this layer: [B, num_heads, N, N]
        if isinstance(attentions, list):
             attn = attentions[layer_idx]
        else:
             # In case it's a stacked tensor
             attn = attentions[layer_idx]

        # Check shape
        if attn.ndim == 3:
            attn = attn.unsqueeze(0) # [1, H, N, N]
        
        # Average heads: [B, N, N] -> [N, N]
        attn_map = attn.mean(dim=1)[0]
        
        # 1. Inter-patch weights (subset of full matrix)
        plt.subplot(5, num_layers, i + 1)
        # Slicing [1:, 1:] removes CLS token interaction
        # We visualize the average attention to *all* tokens from *all* tokens? 
        # Actually usually people visualize "Attention TO x" or "Attention FROM x".
        # The original code did: plt.imshow(attn_map[1:, 1:]).
        # That's a 196x196 matrix. It will look like a mess or a diagonal line.
        # But let's keep original behavior for now.
        plt.imshow(attn_map[1:, 1:].float().cpu().numpy(), cmap='viridis')
        plt.title(f'Layer {layer_idx}\nInter-Patch Matrix')
        plt.axis('off')

        # 2. CLS token norm visualization (Attention FROM CLS to Patches)
        # attn_map[0, 1:] -> row 0 is CLS attending to patches
        cls_attn = attn_map[0, 1:]
        cls_attn_grid = cls_attn.reshape(h, w)
        
        plt.subplot(5, num_layers, i + 1 * num_layers + 1)
        plt.imshow(cls_attn_grid.float().cpu().numpy(), cmap='viridis')
        plt.title(f'CLS Attn (Raw)')
        plt.axis('off')
        
        # 3. CLS token overlay
        plt.subplot(5, num_layers, i + 2 * num_layers + 1)
        # Normalize for visualization
        cls_attn_norm = (cls_attn_grid - cls_attn_grid.min()) / (cls_attn_grid.max() - cls_attn_grid.min() + 1e-6)
        
        cls_attn_resized = F.interpolate(
            cls_attn_norm.unsqueeze(0).unsqueeze(0).float(), 
            size=(224, 224), mode='bilinear', align_corners=False
        )[0, 0]
        plt.imshow(img_vis)
        plt.imshow(cls_attn_resized.cpu().numpy(), alpha=0.5, cmap='jet')
        plt.title(f'CLS Attn Overlay')
        plt.axis('off')

        # 4. Selected patch self-attention (Center Patch attending to others)
        # attn_map[center_idx, 1:]
        patch_attn = attn_map[center_idx, 1:]
        patch_attn_grid = patch_attn.reshape(h, w)
        
        plt.subplot(5, num_layers, i + 3 * num_layers + 1)
        plt.imshow(patch_attn_grid.float().cpu().numpy(), cmap='viridis')
        plt.title(f'Patch ({center_h},{center_w}) Attn')
        plt.axis('off')

        # 5. Selected patch overlay
        plt.subplot(5, num_layers, i + 4 * num_layers + 1)
        
        patch_attn_norm = (patch_attn_grid - patch_attn_grid.min()) / (patch_attn_grid.max() - patch_attn_grid.min() + 1e-6)

        patch_attn_resized = F.interpolate(
            patch_attn_norm.unsqueeze(0).unsqueeze(0).float(),
            size=(224, 224), mode='bilinear', align_corners=False
        )[0, 0]
        plt.imshow(patch_attn_resized.cpu().numpy(), alpha=0.5, cmap='jet')
        
        # Draw red dot at center of selected patch
        # h, w = 14, 14. patch_size = 16.
        # center of patch (h, w) in pixels is (w*16 + 8, h*16 + 8)
        row_px = center_h * patch_size + patch_size // 2
        col_px = center_w * patch_size + patch_size // 2
        plt.plot(col_px, row_px, marker='+', color='white', markersize=12, markeredgewidth=2)
        plt.title(f'Patch Attn Overlay')
        plt.axis('off')

    plt.tight_layout()
    output_path = 'attention_visualization_grid.png'
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")

    # --- Segmentation Visualization ---
    print("Running segmentation inference...")
    # Setup for forward_feature
    # SegmentorEx expects tensor input for forward_feature
    
    # We need to make sure we don't run out of memory or conflict with previous gradients
    # But we are in no_grad context in main? No, previous no_grad was inside get_attention_maps
    
    with torch.no_grad():
        # transform already resizes to 224x224
        # model.forward_feature expects a tensor [B, 3, H, W]
        # and returns logits [B, NumLines, H, W]
        
        # The 'outputs' in get_attention_maps used model.net.encode_image directly.
        # Now we use the high-level SegmentorEx method.
        # Note: forward_feature might re-run encode_image.
        
        # img was pre-processed inside get_attention_maps but not returned.
        # Let's re-process here or grab it from somewhere.
        # We can just re-create the tensor locally.
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
        ])
        seg_input = transform(img).unsqueeze(0).to(device)
        if hasattr(model.net.visual, 'conv1') and model.net.visual.conv1.weight.dtype == torch.float16:
            seg_input = seg_input.half()

        logits = model.forward_feature(seg_input)
        # logits shape: [1, NumClasses, H, W]
        
        preds = torch.argmax(logits, dim=1)[0] # [H, W]
        preds_np = preds.cpu().numpy().astype(np.uint8)
        
    # Visualize Segmentation
    plt.figure(figsize=(10, 5))
    
    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(img.resize((224, 224)))
    plt.title("Input Image")
    plt.axis('off')
    
    # Segmentation Mask
    plt.subplot(1, 2, 2)
    # Use a colorful map for classes
    plt.imshow(preds_np, cmap='jet', vmin=0, vmax=model.num_classes-1)
    plt.title("Segmentation Prediction")
    plt.axis('off')
    
    seg_output_path = 'segmentation_result.png'
    plt.savefig(seg_output_path)
    print(f"Saved segmentation result to {seg_output_path}")
    
    plt.show() # Show both windows (matplotlib handles multiple figures usually, or showing the last one)
    # Existing plt.show() was for the attention grid.
    # To show both, we can comment out the first show() or rely on non-blocking behavior.
    # But usually plt.show() blocks.
    # Let's combine them into one show() call at the end, but they are separate figures.
    # Matplotlib show() displays all open figures.

def main():
    print("Initializing Model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SegmentorEx(
        clip_type='CLIP',
        vit_type='ViT-B/16',
        model_type='SegEarth',
        name_path='configs/cls_potsdam.txt',
        device=device,
        ignore_residual=True,
        apply_sim_feat_up=False, 
        apply_outlier_suppression=False,
        outlier_suppression_cfg=dict(top_k=30),
        apply_similarity_enhancement=False,
        similarity_enhancement_cfg=dict(similarity_weight=1.0, temperature=1.0, add_self_similarity=True),
        sim_feat_up_cfg=dict(model_name='jbu_one', model_path='simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt')
    )
    model.eval()
    img = Image.open(img_path)
        
    visualize_layers(model, img, device)

if __name__ == "__main__":
    main()
