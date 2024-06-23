auto_scale_lr = dict(base_batch_size=1024)
data_preprocessor = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
bgr_mean = [103.53, 116.28, 123.675]
bgr_std = [57.375, 57.12, 58.395]

default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='VisualizationHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = '/mnt/pfs-mc0p4k/cv/team/panxuhao/playground/miccai24_cxr_lt/code/train/work_dirs/mae_vit-large-p16_8xb512-1600e_in37w/epoch_1600.pth'
log_level = 'INFO'
model = dict(
    backbone=dict(
        arch='l',
        drop_rate=0.1,
        img_size=448,
        patch_size=16,
        type='VisionTransformer'),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=40,
        in_channels=1024,
        loss=dict(type='FocalLoss')),
    neck=None,
    type='ImageClassifier')
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=0.0001,
        type='AdamW',
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict({
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0)
        })))
param_scheduler = [
    dict(
        by_epoch=True,
        convert_to_iter_based=True,
        end=5,
        start_factor=0.001,
        type='LinearLR'),
    dict(begin=5, by_epoch=True, eta_min=1e-05, type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, seed=None)
resume = False
train_cfg = dict(by_epoch=True, max_epochs=15, val_interval=1)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=448,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs')
]
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        type='MultiLabelDataset',
        ann_file='/mnt/pfs-mc0p4k/cv/team/panxuhao/playground/miccai24_cxr_lt/code/dataprocesser/data_unbalanced_metainfo.json',
        pipeline=train_pipeline),
    num_workers=5,
    sampler=dict(shuffle=True, type='DefaultSampler'))

val_cfg = dict()
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=448,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=448),
    dict(type='PackInputs')
]
val_dataloader = dict(
    batch_size=16,
    dataset=dict(
        type='MultiLabelDataset',
        ann_file='/mnt/pfs-mc0p4k/cv/team/panxuhao/playground/miccai24_cxr_lt/code/dataprocesser/val_data.json',
        pipeline=val_pipeline),
    num_workers=5,
    sampler=dict(shuffle=True, type='DefaultSampler'))

val_evaluator = dict(
    type='AveragePrecision')

vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])

test_cfg = dict()
test_pipeline = val_pipeline
test_dataloader = val_dataloader
test_evaluator = dict(
    type='AveragePrecision')

work_dir = '/mnt/pfs-mc0p4k/cv/team/panxuhao/playground/miccai24_cxr_lt/code/train/work_dirs/mae_vit-large-mae-e1600-ft-unbalanced-data_e15'
