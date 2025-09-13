import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

import argparse
import tensorflow as tf
from tqdm import tqdm
from pathlib import Path

tf.compat.v1.enable_eager_execution()

from src.utils import file_utils, Camera, Lidar


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Clear2Fog on Waymo Dataset.")
    parser.add_argument("-i", "--input_dir", type=Path, required=True, help="Input folder containing tfrecords.")
    parser.add_argument("-o", "--output_dir", type=Path, required=True, help="Empty output directory of where to place the fog-simulated files.")
    parser.add_argument("-c", "--camera_ids", type=int, nargs='+', default=[1], help="List of camera IDs to process [1 to 5] - check waymo_open_dataset/dataset.proto")
    args = parser.parse_args()

    print("\n")
    for i, file in enumerate(tqdm(os.listdir(args.input_dir), desc="Processing Waymo files", leave=False), start=1):
        file_path = os.path.join(args.input_dir, file)

        subfolders = file_utils.output_folders_waymo(file, args.output_dir)

        frames = file_utils.read_waymo_frames(file_path, n_frames=10000)  # Select a very high number (e.g. 10000) to read all frames in the scene
        for i, frame in enumerate(tqdm(frames, desc=f"Frames in {file}", leave=False), start=1):          
            for cam_id in args.camera_ids:
                # Save camera images
                cam = Camera(frame, i, subfolders, camera_id=cam_id)
                rgb_image_path = cam.decode_camera_image(labels=False)
                cam.save_camera_image()

                # Save LiDAR point clouds
                lid = Lidar(frame, i, subfolders)
                point_cloud, bbox_3d = lid.get_pc_intensity_and_bboxes()  # point_cloud = [x, y, z, intensity]
                lid.save_point_cloud_and_3d_bbox()
