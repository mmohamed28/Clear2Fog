_base_ = '../../faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=3)
    )
)

data_root_train = '/users/sgmmoha5/scratch/waymo_foggy_mixed/'
data_root_holdout = '/users/sgmmoha5/scratch/waymo_foggy_mixed/'
data_root_val = '/data/users/sgmmoha5/waymo_extracted/'
dataset_type = 'CocoDataset'

classes = ('TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST')

train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root_train,
        metainfo=dict(classes=classes),
        ann_file='training/annotations_10.json',
        data_prefix=dict(img='training/foggy_camera/')))

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

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

work_dir = '/mnt/scratch/users/sgmmoha5/mmdetection_checkpoints/mixed/mixed_10_seed3'

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,        
        save_best='auto',
        rule='greater'
    ))

randomness = dict(seed=56)