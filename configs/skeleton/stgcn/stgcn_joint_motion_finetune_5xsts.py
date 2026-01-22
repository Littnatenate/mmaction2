# --- Point to the pre-trained model ---
# MODIFIED: Point to the JOINT-MOTION model checkpoint you downloaded
load_from = 'checkpoints/stgcn_8xb16-joint-motion-u100-80e_ntu120-xsub-keypoint-3d_20221129-5d54f525.pth' 

# --- Model Definition ---
model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        graph_cfg=dict(layout='mediapipe', mode='stgcn_spatial')), # MODIFIED: layout='mediapipe'
    cls_head=dict(
        type='GCNHead',
        num_classes=2,  # MODIFIED: 2 Classes (Healthy vs LBP)
        in_channels=256))

# --- Dataset Settings ---
# ADDED: Your custom dataset settings
dataset_type = 'PoseDataset'
data_root = 'data/5xSTS'
ann_file_train = f'{data_root}/train_annotations.pkl'
ann_file_val = f'{data_root}/val_annotations.pkl'

# --- Data Pipelines ---
# MODIFIED: Pipelines have been simplified and corrected for your custom data
train_pipeline = [
    dict(type='GenSkeFeat', feats=['jm']),          # MODIFIED: Set to 'jm' for JOINT-MOTION
    dict(type='UniformSampleFrames', clip_len=100),
    dict(type='FormatGCNInput', num_person=1),      # MODIFIED: Set to 1 person for your data
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='GenSkeFeat', feats=['jm']),          # MODIFIED: Set to 'jm' for JOINT-MOTION
    dict(type='UniformSampleFrames', clip_len=100, num_clips=1, test_mode=True),
    dict(type='FormatGCNInput', num_person=1),      # MODIFIED: Set to 1 person
    dict(type='PackActionInputs')
]
test_pipeline = val_pipeline

# --- Dataloaders ---
# MODIFIED: Dataloaders now point to your custom data and pipelines
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
            ann_file=ann_file_train,  # MODIFIED: Points to your training data
            pipeline=train_pipeline
            )))

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,    # MODIFIED: Points to your validation data
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = val_dataloader

# --- Evaluator ---
val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

# --- Training Schedule & Optimizer ---
# MODIFIED: Training schedule set for your fine-tuning run
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=0,
        T_max=50, # MODIFIED: Matched to max_epochs
        by_epoch=True,
        convert_to_iter_based=True)
]

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005))

# --- Hooks & Runtime Settings ---
default_scope = 'mmaction'
# MODIFIED: Hooks configured for your run
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
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)

# ADDED: W&B Visualizer for experiment tracking
visualizer = dict(
    type='ActionVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(project='5xSTS-Joint-Motion-Model')) # ADDED: New W&B project name
    ])

# python tools/train.py configs/skeleton/stgcn/stgcn_joint_motion_finetune_5xsts.py