import os
_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_xBD.txt',
    prob_thd=0.0,
)

# dataset settings
dataset_type = 'xBDDataset'
data_root = os.path.abspath('payload/datasets/xBD')

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
            img_path=f'{data_root}/test_images_labels_targets/images',
            seg_map_path=f'{data_root}/test_images_labels_targets/targets'),
        pipeline=test_pipeline))
