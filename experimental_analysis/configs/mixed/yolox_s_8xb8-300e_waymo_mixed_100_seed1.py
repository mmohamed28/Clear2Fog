_base_ = '../../yolox/yolox_s_8xb8-300e_coco.py'

model = dict(
    bbox_head=dict(num_classes=3))

data_root_train = '/users/sgmmoha5/scratch/waymo_foggy_mixed/'
data_root_holdout = '/users/sgmmoha5/scratch/waymo_foggy_mixed/'
data_root_val = '/data/users/sgmmoha5/waymo_extracted/'
dataset_type = 'CocoDataset'

classes = ('TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST')

train_dataset = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root_train,
        metainfo=dict(classes=classes),
        ann_file='training/annotations_100.json',
        data_prefix=dict(img='training/foggy_camera/')))

train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=train_dataset)

val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root_holdout,
        metainfo=dict(classes=classes),
        ann_file='hold_out/annotations_holdout.json',
        data_prefix=dict(img='hold_out/foggy_camera/')))

test_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root_val,
        metainfo=dict(classes=classes),
        ann_file='validation/annotations.json',
        data_prefix=dict(img='validation/images/')))

val_evaluator = dict(
    ann_file=data_root_holdout + 'hold_out/annotations_holdout.json')

test_evaluator = dict(
    ann_file=data_root_val + 'validation/annotations.json')

max_epochs = 12
train_cfg = dict(max_epochs=max_epochs, val_interval=1)

param_scheduler = [
    dict(
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=5,
        end=max_epochs,
        T_max=max_epochs - 5,
        convert_to_iter_based=True)
]

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'

work_dir = '/mnt/scratch/users/sgmmoha5/mmdetection_checkpoints/mixed/yolox_mixed_100_seed1'

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,        
        save_best='auto',
        rule='greater'
    ))

randomness = dict(seed=12)
