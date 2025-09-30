# --- Point to the pre-trained model ---
load_from = 'checkpoints/stgcn_8xb16-bone-u100-80e_ntu120-xsub-keypoint-3d_20221129-bc007510.pth'

# --- Model Definition (ST-GCN for Bone) ---
model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        graph_cfg=dict(layout='nturgb+d', mode='spatial')),
    cls_head=dict(
        type='GCNHead',
        num_classes=8,  # Your 8 classes
        in_channels=256))

# --- Dataset Settings ---
dataset_type = 'PoseDataset'
data_root = 'data/5xSTS'
ann_file_train = f'{data_root}/train_annotations.pkl'
ann_file_val = f'{data_root}/val_annotations.pkl'

# --- Data Pipelines ---
train_pipeline = [
    dict(type='GenSkeFeat', feats=['b']),
    dict(type='UniformSampleFrames', clip_len=100),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='GenSkeFeat', feats=['b']),
    dict(type='UniformSampleFrames', clip_len=100, num_clips=1, test_mode=True),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]
test_pipeline = val_pipeline

# --- Dataloaders ---
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=15,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file_train,
            pipeline=train_pipeline)))

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        test_mode=True))

# EXPLICITLY define test_dataloader to avoid base file issues
test_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=test_pipeline,
        test_mode=True))

# --- Evaluator ---
val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

# --- Training Schedule & Optimizer ---
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=0,
        T_max=50,
        by_epoch=True,
        convert_to_iter_based=True)
]

optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005))

# --- Hooks & Runtime Settings ---
default_scope = 'mmaction'
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, ignore_last=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', interval=5, save_best='acc/top1'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)
visualizer = dict(
    type='ActionVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(project='5xSTS-Bone-Model'))
    ])

# python tools/train.py configs/skeleton/stgcn/stgcn_bone_finetune_5xsts.py


"""
python tools/test.py `
    configs/skeleton/stgcn/stgcn_bone_finetune_5xsts.py `
    work_dirs/stgcn_bone_finetune_5xsts/best_acc_top1_epoch_XX.pth # <-- Use your best .pth file here

"""