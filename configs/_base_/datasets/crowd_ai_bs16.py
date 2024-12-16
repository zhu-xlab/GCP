# dataset settings
dataset_type = 'CrowdAIDataset'
datapipe = 'crowd_ai'
data_root = '../../Datasets/Dataset4EO/CrowdAI'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
backend_args = None
crop_size = (320, 320)

batch_augments = [
    dict(
        type='BatchFixedSizePad',
        size=crop_size,
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=True,
        seg_pad_value=255)
]
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=False,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=True,
    seg_pad_value=255,
    batch_augments=batch_augments
)


train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False, with_poly_json=False),
    # dict(type='ErodeGT', kernel_size=5),
    # dict(type='Resize', img_scale=(320, 320), ratio_range=(0.8, 1.25)),
    dict(type='Resize', scale=(320, 320), keep_ratio=True),
    # dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='Rotate90', prob=0.75),
    # dict(type='CropFeaturesToBounds'),
    # dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0),
    # dict(type='DefaultFormatBundle'),
    # dict(type='Collect', keys=['img', 'gt_semantic_seg', 'eroded_gt_semantic_seg'], cpu_keys=['features']),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(320, 320), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=False, with_mask=True, poly2mask=False, with_poly_json=False),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    # persistent_workers=True,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=train_pipeline,
        backend_args=backend_args,
        # datapipe=datapipe,
        # split='train'
        ann_file='8e089a94-555c-4d7b-8f2f-4d733aebb058_train/train/annotation.json',
        data_prefix=dict(img='8e089a94-555c-4d7b-8f2f-4d733aebb058_train/train/images'),
    )
)

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    # persistent_workers=True,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        backend_args=backend_args,

        ann_file='0a5c561f-e361-4e9b-a3e2-94f42a003a2b_val/val/annotation-small.json',
        # ann_file='0a5c561f-e361-4e9b-a3e2-94f42a003a2b_val/val/annotation.json',
        data_prefix=dict(img='0a5c561f-e361-4e9b-a3e2-94f42a003a2b_val/val/images'),
        test_mode=True,
        # datapipe=datapipe,
        # split='val_small',
        # coco_ann_path = '../../Datasets/Dataset4EO/CrowdAI/0a5c561f-e361-4e9b-a3e2-94f42a003a2b_val/val/annotation.json',
    )
)
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    # persistent_workers=True,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        backend_args=backend_args,

        ann_file='0a5c561f-e361-4e9b-a3e2-94f42a003a2b_val/val/annotation-small.json',
        # ann_file='0a5c561f-e361-4e9b-a3e2-94f42a003a2b_val/val/annotation.json',
        data_prefix=dict(img='0a5c561f-e361-4e9b-a3e2-94f42a003a2b_val/val/images'),
        test_mode=True,
        # datapipe=datapipe,
        # split='val_small',
        # coco_ann_path = '../../Datasets/Dataset4EO/CrowdAI/0a5c561f-e361-4e9b-a3e2-94f42a003a2b_val/val/annotation.json',
    )
)
