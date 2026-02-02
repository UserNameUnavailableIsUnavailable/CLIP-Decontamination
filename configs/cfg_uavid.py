import os

_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_uavid.txt',
    prob_thd=0.3,
)

# dataset settings
dataset_type = 'UAVidDataset'
data_root = 'UAVid'
data_root = os.path.abspath('payload/datasets/UAVid')

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
            img_path=f'{data_root}/images/validation',
            seg_map_path=f'{data_root}/annotations/validation'),
        pipeline=test_pipeline))