import os
import json
import pathlib
import argparse
import datetime

import tensorflow as tf

from waymo_open_dataset import dataset_pb2 as open_dataset

tf.compat.v1.enable_eager_execution()


class WaymoCOCOConverter():
    def __init__(self,
                 image_dir,
                 image_prefix=None,
                 write_image=True,
                 add_waymo_info=False,
                 add_coco_info=True,
                 frame_index_ones_place=None):
        """
        Parameters
        ----------
        add_waymo_info : bool
            include additional information out of original COCO format.
        add_coco_info : bool
            include information in original COCO format,
            but out of Waymo Open Dataset.
            if set to False, COCO compatibility breaks.
        frame_index_ones_place : int
            extract 1/10 size dataset based on ones place of frame index.
        """

        self.image_dir = image_dir
        self.image_prefix = image_prefix
        self.write_image = write_image
        self.add_waymo_info = add_waymo_info
        self.add_coco_info = add_coco_info
        if frame_index_ones_place is not None:
            self.frame_index_ones_place = int(frame_index_ones_place)
            assert 0 <= self.frame_index_ones_place < 10
        else:
            self.frame_index_ones_place = None

        self.init_waymo_dataset_proto_info()
        self.init_coco_format_info()

        self.img_index = 0
        self.annotation_index = 0
        self.img_dicts = []
        self.annotation_dicts = []

    def init_waymo_dataset_proto_info(self):
        self.waymo_class_mapping = [
            'TYPE_UNKNOWN', 'TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_SIGN',
            'TYPE_CYCLIST'
        ]
        self.waymo_camera_names = {
            0: 'UNKNOWN',
            1: 'FRONT',
            2: 'FRONT_LEFT',
            3: 'FRONT_RIGHT',
            4: 'SIDE_LEFT',
            5: 'SIDE_RIGHT',
        }

    def init_coco_format_info(self):
        self.dataset_info = {
            "year": 2024,
            "version": "v1.4.3",
            "description": "Waymo Open Dataset 2D Detection",
            "contributor": "Waymo LLC",
            "url": "https://waymo.com/open/",
            "date_created": datetime.datetime.utcnow().isoformat(' '),
        }
        self.licenses = [{
            "id": 1,
            "name": "Waymo Dataset License Agreement for Non-Commercial Use",
            "url": "https://waymo.com/open/terms/",
        }]
        self.target_categories = [
            {
                "id": 1,
                "name": "TYPE_VEHICLE",
                "supercategory": "vehicle",
            },
            {
                "id": 2,
                "name": "TYPE_PEDESTRIAN",
                "supercategory": "person",
            },
            {
                "id": 3,
                "name": "TYPE_CYCLIST",
                "supercategory": "bike_plus",
            },
        ]

    def process_sequences(self, tfrecord_paths):
        if not isinstance(tfrecord_paths, (list, tuple)):
            tfrecord_paths = [tfrecord_paths]

        for tfrecord_index, tfrecord_path in enumerate(sorted(tfrecord_paths)):
            sequence_data = tf.data.TFRecordDataset(str(tfrecord_path),
                                                   compression_type='')
            for frame_index, frame_data in enumerate(sequence_data):
                if self.frame_index_ones_place is not None:
                    if frame_index % 10 != self.frame_index_ones_place:
                        continue
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(frame_data.numpy()))
                self.process_frame(frame, frame_index, tfrecord_index)

    def process_frame(self, frame, frame_index, tfrecord_index):
        for camera_image in frame.images:
            self.process_img(camera_image, frame, frame_index, tfrecord_index)
            for camera_label in frame.camera_labels:
                if camera_label.name != camera_image.name:
                    continue
                self.add_coco_annotation_dict(camera_label)
            self.img_index += 1

    def process_img(self, camera_image, frame, frame_index, tfrecord_index):
        img_filename = f'{pathlib.Path(frame.context.name).stem}_{frame_index:03d}_{self.waymo_camera_names[camera_image.name]}.jpg'
        if self.image_prefix is not None:
            img_filename = self.image_prefix + '_' + img_filename
        img_path = os.path.join(self.image_dir, img_filename)

        img = tf.image.decode_jpeg(camera_image.image).numpy()
        img_height = img.shape[0]
        img_width = img.shape[1]
        if self.write_image:
            with open(img_path, 'wb') as f:
                f.write(bytearray(camera_image.image))

        self.add_coco_img_dict(img_filename,
                               height=img_height,
                               width=img_width,
                               sequence_id=tfrecord_index,
                               frame_id=frame_index,
                               camera_id=int(camera_image.name),
                               frame=frame)

    def add_coco_img_dict(self, file_name, height=None, width=None, sequence_id=None, frame_id=None, camera_id=None, frame=None):
        if height is None or width is None:
            raise ValueError
        img_dict = {
            "id": self.img_index,
            "width": width,
            "height": height,
            "file_name": file_name,
            "license": 1,
        }
        if self.add_coco_info:
            img_dict["flickr_url"] = ""
            img_dict["coco_url"] = ""
            img_dict["date_captured"] = ""
        if self.add_waymo_info:
            img_dict["context_name"] = frame.context.name
            img_dict["timestamp_micros"] = frame.timestamp_micros
            img_dict["camera_id"] = camera_id
            img_dict["sequence_id"] = sequence_id
            img_dict["frame_id"] = frame_id
            img_dict["time_of_day"] = frame.context.stats.time_of_day
            img_dict["location"] = frame.context.stats.location
            img_dict["weather"] = frame.context.stats.weather
        self.img_dicts.append(img_dict)

    def add_coco_annotation_dict(self, camera_label):
        annotation_dicts = []
        for box_label in camera_label.labels:
            category_name_to_id = {
                category['name']: category['id']
                for category in self.target_categories
            }
            category_name = self.waymo_class_mapping[box_label.type]
            if category_name not in category_name_to_id:
                continue
            category_id = category_name_to_id[category_name]
            width = box_label.box.length
            height = box_label.box.width
            x1 = box_label.box.center_x - width / 2
            y1 = box_label.box.center_y - height / 2
            annotation_dict = {
                "id": self.annotation_index,
                "image_id": self.img_index,
                "category_id": category_id,
                "segmentation": [],
                "area": width * height,
                "bbox": [x1, y1, width, height],
                "iscrowd": 0,
            }
            if self.add_waymo_info:
                annotation_dict["track_id"] = box_label.id
            annotation_dicts.append(annotation_dict)
            self.annotation_index += 1
        self.annotation_dicts.extend(annotation_dicts)

    def write_coco(self, label_path, json_indent=None):
        output_dict = {
            "info": self.dataset_info,
            "licenses": self.licenses,
            "categories": self.target_categories,
            "images": self.img_dicts,
        }
        if self.annotation_dicts:
            output_dict["annotations"] = self.annotation_dicts
        if self.add_waymo_info:
            output_dict["camera_names"] = self.waymo_camera_names

        with open(label_path, mode='w') as f:
            json.dump(json.loads(json.dumps(output_dict),
                                parse_float=lambda x: round(float(x), 6)),
                      f,
                      indent=json_indent,
                      sort_keys=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecord_file', required=True, type=str, help="Path to a single .tfrecord file.")
    parser.add_argument('--work_dir', required=True, type=str, help="Base output directory for the split.")
    parser.add_argument('--image_dirname', default='images', type=str)
    parser.add_argument('--add_waymo_info', action='store_true')
    parser.add_argument('--skip_write_image', action='store_true')
    args = parser.parse_args()

    image_dir = os.path.join(args.work_dir, args.image_dirname)

    temp_label_dir = os.path.join(args.work_dir, "annotations_temp")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(temp_label_dir, exist_ok=True)
    
    tfrecord_basename = os.path.splitext(os.path.basename(args.tfrecord_file))[0]
    label_path = os.path.join(temp_label_dir, f"{tfrecord_basename}.json")

    tfrecord_list = [args.tfrecord_file]

    waymo_converter = WaymoCOCOConverter(
        image_dir,
        write_image=(not args.skip_write_image),
        add_waymo_info=args.add_waymo_info)

    waymo_converter.process_sequences(tfrecord_list)

    waymo_converter.write_coco(label_path, json_indent=None)

    print(f"Finished converting {args.tfrecord_file}")
    
    print(f"DELETING source file: {args.tfrecord_file}")
    os.remove(args.tfrecord_file)


if __name__ == "__main__":
    main()
