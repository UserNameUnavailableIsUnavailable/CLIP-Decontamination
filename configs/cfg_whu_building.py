import os
_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_whu.txt',
    prob_thd=0.6,
)

# dataset settings
dataset_type = 'WHUDataset'
data_root = os.path.abspath('payload/datasets/WHU-Building')

test_pipeline = [
    dict(type='LoadImageFromFile'),
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
            img_path=f'{data_root}/images/test',
            seg_map_path=f'{data_root}/annotations/test'),
        pipeline=test_pipeline))
