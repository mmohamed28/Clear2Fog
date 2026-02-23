_base_ = '../../faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=3)
    )
)

data_root_train = '/users/sgmmoha5/data/STF_COCO/split/'
data_root_holdout = '/users/sgmmoha5/data/STF_COCO/split/'
data_root_val = '/users/sgmmoha5/data/STF_COCO/split/'
dataset_type = 'CocoDataset'

classes = ('TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST')

train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root_train,
        metainfo=dict(classes=classes),
        ann_file='train/stf_annotations_coco_train_remap.json',
        data_prefix=dict(img='train/images/')))

val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root_holdout,
        metainfo=dict(classes=classes),
        ann_file='val/stf_annotations_coco_val_remap.json',
        data_prefix=dict(img='val/images/')))

test_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root_val,
        metainfo=dict(classes=classes),
        ann_file='test/stf_annotations_coco_test_remap.json',
        data_prefix=dict(img='test/images/')))

val_evaluator = dict(
    ann_file=data_root_holdout + 'val/stf_annotations_coco_val_remap.json')

test_evaluator = dict(
    ann_file=data_root_val + 'test/stf_annotations_coco_test_remap.json')

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

work_dir = '/mnt/scratch/users/sgmmoha5/mmdetection_checkpoints/stf/foggy_seed2'

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,        
        save_best='auto',
        rule='greater'
    ))

randomness = dict(seed=34)