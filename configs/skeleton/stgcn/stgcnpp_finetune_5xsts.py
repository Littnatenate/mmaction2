
custom_imports = dict(imports=['my_transform'], allow_failed_imports=False)

_base_ = ['../../_base_/default_runtime.py']

# Point to pre-trained checkpoint (adjust path if you downloaded elsewhere)
load_from = 'checkpoints/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221228-86e1e77a.pth'

# Model definition
model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        graph_cfg=dict(layout='nturgb+d', mode='spatial')  # <-- use COCO since your data is from Mediapipe
    ),
    cls_head=dict(
        type='GCNHead',
        num_classes=8,   # 8 action classes in 5xSTS
        in_channels=256)
)

# Dataset settings
dataset_type = 'PoseDataset'
data_root = 'data/5xSTS'
ann_file_train = f'{data_root}/train_annotations.pkl'
ann_file_val = f'{data_root}/val_annotations.pkl'

# Pipelines
# Replace your old train_pipeline with this one
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=80),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=80, num_clips=1, test_mode=True),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]
test_pipeline = val_pipeline

# Data loaders
train_dataloader = dict(
    batch_size=8,
    num_workers=4,

    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=15,
        dataset=dict(
            type=dataset_type,          # e.g. 'PoseDataset'
            ann_file=ann_file_train,    # path to train annotation file
            pipeline=train_pipeline
        )
    )
)

val_dataloader = dict(
    batch_size=8,
    num_workers=4,

    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        test_mode=True
    )
)
test_dataloader = val_dataloader

# Evaluators
val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

# Training loop
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer & LR scheduler
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=0,
        T_max=50,
        by_epoch=True,
        convert_to_iter_based=True
    )
]
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005, nesterov=True),
)

# Runtime settings
default_scope = 'mmaction'
default_hooks = dict(
    checkpoint=dict(interval=5, save_best='acc/top1'),
    logger=dict(interval=1)
)

visualizer = dict(
    type='ActionVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(
                project='5xSTS-pre-trained-train-01'  # Or any name you want for your W&B project
            ))
    ])

env_cfg = dict(cudnn_benchmark=True)
log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)

# python tools/train.py configs/skeleton/stgcn/stgcnpp_finetune_5xsts.py
