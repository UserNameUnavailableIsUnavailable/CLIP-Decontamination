import os
_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_udd5.txt',
    prob_thd=0.4,
    bg_idx=4,
)

# dataset settings
dataset_type = 'UDD5Dataset'
data_root = os.path.abspath('payload/datasets/UDD/UDD5')

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path=f'{data_root}/val/src',
            seg_map_path=f'{data_root}/val/gt'),
        pipeline=test_pipeline))