_base_ = [
    'mmdet::_base_/models/mask-rcnn_r50_fpn.py',
    'mmdet::_base_/datasets/crowd_ai.py',
    # 'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]

# please install the mmpretrain
# import mmpretrain.models to trigger register_module in mmpretrain
custom_imports = dict(
    imports=['mmpretrain.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-base_3rdparty-fcmae_in1k_20230104-8a798eaf.pth'  # noqa

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmpretrain.ConvNeXt',
        arch='base',
        out_indices=[0, 1, 2, 3],
        # TODO: verify stochastic depth rate {0.1, 0.2, 0.3, 0.4}
        drop_path_rate=0.4,
        layer_scale_init_value=0.,  # disable layer scale when using GRN
        gap_before_final_norm=False,
        use_grn=True,  # V2 uses GRN
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    neck=dict(in_channels=[128, 256, 512, 1024]),
    test_cfg=dict(
        rpn=dict(nms=dict(type='nms')),  # TODO: does RPN use soft_nms?
        rcnn=dict(nms=dict(type='soft_nms'))))

max_iters = 80000
interval = 5000
dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=interval,
    dynamic_intervals=dynamic_intervals)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_iters,
        by_epoch=False,
        milestones=[40000, 75000],
        gamma=0.1)
]

# Enable automatic-mixed-precision training with AmpOptimWrapper.
optim_wrapper = dict(
    type='AmpOptimWrapper',
    constructor='LearningRateDecayOptimizerConstructor',
    paramwise_cfg={
        'decay_rate': 0.95,
        'decay_type': 'layer_wise',  # TODO: sweep layer-wise lr decay?
        'num_layers': 12
    },
    optimizer=dict(
        # _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    ))

default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))

val_evaluator = [
    dict(
        type='CocoMetric',
        # ann_file=data_root + 'annotations/instances_val2017.json',
        ann_file='../../Datasets/Dataset4EO/CrowdAI/0a5c561f-e361-4e9b-a3e2-94f42a003a2b_val/val/annotation-small.json',
        # ann_file='../../Datasets/Dataset4EO/CrowdAI/0a5c561f-e361-4e9b-a3e2-94f42a003a2b_val/val/annotation.json',
        metric=['segm'],
        backend_args={{_base_.backend_args}})
]
test_evaluator = val_evaluator
