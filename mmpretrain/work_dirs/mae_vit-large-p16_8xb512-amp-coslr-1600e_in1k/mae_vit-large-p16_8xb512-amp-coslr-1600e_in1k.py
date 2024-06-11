custom_imports = dict(
    allow_failed_imports=False, imports='mmpretrain.datasets')
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ])
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=3, type='CheckpointHook'),
    logger=dict(interval=1, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(
    custom_cfg=[
        dict(data_src='', method='mean', windows_size='global'),
    ],
    window_size=10)
model = dict(
    backbone=dict(arch='l', mask_ratio=0.75, patch_size=16, type='MAEViT'),
    head=dict(
        loss=dict(criterion='L2', type='PixelReconstructionLoss'),
        norm_pix=True,
        patch_size=16,
        type='MAEPretrainHead'),
    init_cfg=dict(
        checkpoint=
        '/mnt/pfs-mc0p4k/cv/team/panxuhao/playground/miccai24_cxr_lt/code/train/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k_20220825-cc7e98c9.pth',
        type='Pretrained'),
    neck=dict(
        decoder_depth=8,
        decoder_embed_dim=512,
        decoder_num_heads=16,
        embed_dim=1024,
        in_chans=3,
        mlp_ratio=4.0,
        patch_size=16,
        type='MAEPretrainDecoder'),
    type='MAE')
optim_wrapper = dict(
    loss_scale='dynamic',
    optimizer=dict(
        betas=(
            0.9,
            0.95,
        ), lr=0.0024, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict(
            bias=dict(decay_mult=0.0),
            cls_token=dict(decay_mult=0.0),
            ln=dict(decay_mult=0.0),
            mask_token=dict(decay_mult=0.0),
            pos_embed=dict(decay_mult=0.0))),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=40,
        start_factor=0.0001,
        type='LinearLR'),
    dict(
        T_max=1560,
        begin=40,
        by_epoch=True,
        convert_to_iter_based=True,
        end=1600,
        type='CosineAnnealingLR'),
]
randomness = dict(diff_rank_seed=True, seed=0)
resume = False
train_cfg = dict(max_epochs=1600, type='EpochBasedTrainLoop')
train_dataloader = dict(
    batch_size=256,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file=
        '/mnt/pfs-mc0p4k/cv/team/panxuhao/playground/miccai24_cxr_lt/code/dataprocesser/anno.json',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                crop_ratio_range=(
                    0.2,
                    1.0,
                ),
                interpolation='bicubic',
                scale=224,
                type='RandomResizedCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
        type='Jsonlist'),
    num_workers=16,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        backend='pillow',
        crop_ratio_range=(
            0.2,
            1.0,
        ),
        interpolation='bicubic',
        scale=224,
        type='RandomResizedCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/mae_vit-large-p16_8xb512-amp-coslr-1600e_in1k'
