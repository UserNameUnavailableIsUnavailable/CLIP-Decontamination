# Configuration example with Cross-Tile Fusion enabled

_base_ = './base_config.py'

# model settings with cross-tile fusion
model = dict(
    type='SegEarthSegmentation',
    clip_type='CLIP',
    vit_type='ViT-B/16',
    model_type='SegEarth',
    ignore_residual=True,
    apply_sim_feat_up=True,
    sim_feat_up_cfg=dict(
        model_name='jbu_one',
        model_path='simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt'),
    cls_token_lambda=-0.3,
    
    # Cross-Tile Fusion settings
    apply_cross_tile_fusion=True,
    cross_tile_fusion_mode='weighted',  # 'weighted' (similarity-based) or 'attention' (self-attention)
    cross_tile_fusion_strength=0.5,     # Blending strength (0=no fusion, 1=full fusion)
    
    # Optional: Enable SOM for outlier suppression
    apply_cos=False,
    
    # Optional: Enable CTD for clustering-based debiasing
    apply_ctd=False,
)

# Dataset can be any dataset that uses sliding window inference
# Example for Potsdam:
dataset_type = 'PotsdamDataset'
data_root = ''

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(448, 448), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='data/potsdam/img_dir/val',
            seg_map_path='data/potsdam/ann_dir/val'),
        pipeline=test_pipeline))
