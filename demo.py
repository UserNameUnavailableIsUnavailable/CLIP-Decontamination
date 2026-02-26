from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from segmentor import SegmentorEx

img = Image.open('visualization/img.png').convert('RGB')
dataset = "UDD5"
cls = []
with open(f'configs/cls_{dataset.lower()}.txt', 'r') as f:
    for line in f:
        cls.append(line.strip())
print(f"Classes: {cls}")

img_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    transforms.Resize((448, 448))
])(img)

img_tensor = img_tensor.unsqueeze(0).to('cuda')

model = SegmentorEx(
    clip_type='CLIP',     # 'CLIP', 'BLIP', 'OpenCLIP', 'MetaCLIP', 'ALIP', 'SkyCLIP', 'GeoRSCLIP', 'RemoteCLIP'
    vit_type='ViT-B/16',      # 'ViT-B/16', 'ViT-L-14'
    model_type='SegEarth',   # 'vanilla', 'MaskCLIP', 'GEM', 'SCLIP', 'ClearCLIP', 'SegEarth', 'Combined_Experimental'
    ignore_residual=True,
    apply_sim_feat_up=True,
    sim_feat_up_cfg=dict(
        model_name='jbu_one',
        model_path='simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt'),
    cls_token_lambda=-0.3,
    global_debias_factor=0.5,
    apply_outlier_suppression=True,
    outlier_suppression_cfg=dict(top_k=10),
    apply_similarity_enhancement=True,
    similarity_enhancement_cfg=dict(similarity_weight=1.0, temperature=1.0, add_self_similarity=True),
    name_path=f'configs/cls_{dataset.lower()}.txt',
    prob_thd=0.1,
)


seg_pred = model.predict(img_tensor, data_samples=None)
seg_pred = seg_pred.data.cpu().numpy().squeeze(0)

# Load palette
import custom_datasets
import numpy as np

try:
    dataset_cls = getattr(custom_datasets, f"{dataset}Dataset")
    palette = dataset_cls.METAINFO['palette']
except AttributeError:
    palette = None
    print(f"Warning: Dataset class {dataset}Dataset not found in custom_datasets.py")

fig, ax = plt.subplots(1, figsize=(12, 6))

if palette:
    palette = np.array(palette)
    # Handle ignore index if present (e.g. 255)
    # Assuming standard 0-N class indices
    color_seg = np.zeros((seg_pred.shape[0], seg_pred.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg_pred == label, :] = color
    ax.imshow(color_seg)
else:
    ax.imshow(seg_pred, cmap='viridis')

ax.axis('off')
plt.tight_layout()
# plt.show()
plt.savefig('visualization/result.png', bbox_inches='tight')