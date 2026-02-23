_base_ = '../../faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=3)
    )
)

data_root_base = '/users/sgmmoha5/data/STF_COCO/split/'
dataset_type = 'CocoDataset'

classes = ('TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST')

train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root_base,
        metainfo=dict(classes=classes),
        ann_file='train/stf_annotations_coco_train_remap.json',
        data_prefix=dict(img='train/images/')))

val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root_base,
        metainfo=dict(classes=classes),
        ann_file='val/stf_annotations_coco_val_remap.json',
        data_prefix=dict(img='val/images/')))

test_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root_base,
        metainfo=dict(classes=classes),
        ann_file='test/stf_annotations_coco_test_remap.json',
        data_prefix=dict(img='test/images/')))

val_evaluator = dict(
    ann_file=data_root_base + 'val/stf_annotations_coco_val_remap.json')

test_evaluator = dict(
    ann_file=data_root_base + 'test/stf_annotations_coco_test_remap.json')

load_from = '/users/sgmmoha5/scratch/mmdetection_checkpoints/fixed/fixed_100_seed1/best_coco_bbox_mAP_epoch_9.pth'

work_dir = '/users/sgmmoha5/scratch/mmdetection_checkpoints/stf/fixed_then_stf/100_seed1_lr0.0002'

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0002, momentum=0.9, weight_decay=0.0001))

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='auto',
        rule='greater'
    ))

randomness = dict(seed=12)