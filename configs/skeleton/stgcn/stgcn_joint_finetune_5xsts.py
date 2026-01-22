# --- Point to the pre-trained model ---
load_from = 'checkpoints/stgcn_8xb16-joint-u100-80e_ntu120-xsub-keypoint-3d_20221129-0484f579.pth' # MODIFIED: Point to the JOINT model checkpoint

# --- Model Definition ---
model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        graph_cfg=dict(layout='mediapipe', mode='stgcn_spatial')), # MODIFIED: layout='mediapipe' for 33 joints
    cls_head=dict(
        type='GCNHead',
        num_classes=2,  # MODIFIED: 2 Classes (Healthy vs LBP)
        in_channels=256))

# --- Dataset Settings ---
dataset_type = 'PoseDataset'
data_root = 'data/5xSTS'
ann_file_train = f'{data_root}/train_annotations.pkl'
ann_file_val = f'{data_root}/val_annotations.pkl'

# --- Data Pipelines ---
# MODIFIED: Pipelines have been simplified and corrected for your custom data
train_pipeline = [
    # REMOVED: PreNormalize3D, as the model has its own internal normalization.
    dict(type='GenSkeFeat', feats=['j']),           # MODIFIED: Explicitly set to 'j' for JOINT
    dict(type='UniformSampleFrames', clip_len=100),
    # REMOVED: PoseDecode, as it's not compatible with your pre-processed .pkl files.
    dict(type='FormatGCNInput', num_person=1),      # MODIFIED: Set to 1 person for your data
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='GenSkeFeat', feats=['j']),           # MODIFIED: Explicitly set to 'j' for JOINT
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
            # REMOVED: 'split' key, which is not needed for your data format
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

test_dataloader = dict(
    batch_size=8,  # Using a safe batch size like 8 is a good practice for testing
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,    # Pointing to your validation set for testing
        pipeline=test_pipeline,
        test_mode=True))

# --- Evaluator ---
val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

# --- Training Schedule & Optimizer ---
# MODIFIED: Training schedule set for your fine-tuning run
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
# ADDED: Essential for transfer learning (25 -> 33 joints)
load_from = 'checkpoints/stgcn_8xb16-joint-u100-80e_ntu120-xsub-keypoint-3d_20221129-0484f579.pth' 
# NOTE: The 'load_from' variable is used by the runner. To force 'strict=False', we rely on the runner's default behavior 
# or explicit 'load_checkpoint' calls. In MMAction2 1.x configs, 'load_from' at root usually handles this, 
# but typically 'strict=False' needs to be passed if the runner supports it in the config or via CLI. 
# However, for simplicity in config, we ensure the variable is set.
# Actually, to strictly enforce strict=False, it's often passed in the 'load_checkpoint' function or init_cfg.
# Let's try to add it to the model's init_cfg if supported, or rely on the fact that we changed the layout.

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
    # ADDED: Using AmpOptimWrapper for faster training on GPU
    type='AmpOptimWrapper',
    optimizer=dict(
        type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)) # MODIFIED: Simplified and set a good starting LR

# --- Hooks & Runtime Settings ---
default_scope = 'mmaction'
# MODIFIED: Hooks configured for your run
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, ignore_last=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', interval=5, save_best='acc/top1'), # MODIFIED: Set save_best metric
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

env_cfg = dict(
    cudnn_benchmark=True, # MODIFIED: Set to True for a speed boost
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
            init_kwargs=dict(project='5xSTS-Joint-Model')) # ADDED: New W&B project name
    ])

# python tools/train.py configs/skeleton/stgcn/stgcn_joint_finetune_5xsts.py