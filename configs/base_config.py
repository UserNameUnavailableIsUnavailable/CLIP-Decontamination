# base configurations
model = dict(
    type='SegmentorEx',
    clip_type='CLIP',
    vit_type='ViT-B/16',
    model_type='Experimental',
    ignore_residual=True,
    apply_sim_feat_up=True,
    cls_token_lambda=0.0,
    # proposed
    global_debias_factor=0.2,
    apply_outlier_suppression=True,
    outlier_suppression_cfg=dict(
        top_k=30,
    ),
    apply_similarity_enhancement=True,
    similarity_enhancement_cfg=dict(
        similarity_weight=1.0,  # Weight for self-similarity when adding to attention
        temperature=1.0,  # Temperature for similarity computation
        add_self_similarity=True,  # Include diagonal in similarity map
    ),
    sim_feat_up_cfg=dict(
        model_name='jbu_one',
        model_path='simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt'),
)

test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, alpha=0.5, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', interval=1))
