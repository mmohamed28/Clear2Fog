import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

import argparse
import logging
import warnings
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from src.fog_pipeline import (
    DepthGenerator,
    FogLidar, 
    FogCamera
)
from src.utils.file_utils import organise_output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Clear2Fog on provided images and point clouds.")
    parser.add_argument("-c", "--input_camera_dir", type=Path, required=True, help="Input folder containing the camera files in [.png, .jpg or .jpeg].")
    parser.add_argument("-l", "--input_lidar_dir", type=Path, help="Input folder containing the point cloud files in [.npy].")
    parser.add_argument("-o", "--output_dir", type=Path, required=True, help="Destination of fog-simulated files.")
    parser.add_argument("-v", "--visibility", type=int, default=100, help="Visibility distance in metres, controlling fog density (lower values = denser fog).")
    args = parser.parse_args()

    # Create the top-level output directories
    include_camera = args.input_camera_dir is not None
    include_lidar = args.input_lidar_dir is not None
    camera_output_dir, lidar_output_dir = organise_output(args.output_dir, camera=include_camera, lidar=include_lidar)

    # Initialise models
    print("\n")
    logging.info("Initialising all models...")
    depth = DepthGenerator()
    camera_fog = FogCamera(depth, args.input_camera_dir)
    lidar_fog = FogLidar()
    logging.info("All models initialised.")
    print("\n")

    if include_camera:
        extensions = ('*.png', '*.jpg', '*.jpeg')
        image_paths = []
        
        for ext in extensions:
            image_paths.extend(args.input_camera_dir.glob(ext))

        for image_path in tqdm(image_paths, desc="Camera Fog Simulation", leave=False):
            try:
                image = Image.open(image_path).convert("RGB")
                image_original_np = np.array(image)

                # Generate depth map of the original image
                depth_pred_np = depth.run(image_path)

                # Apply fog simulation to the original image
                foggy_original_np = camera_fog.simulate(image_original_np, image_original_np, depth_pred_np, visibility=args.visibility, min_depth=1000)
                foggy_image_np = foggy_original_np

                # Save the foggy images using the original file name
                foggy_file_name = f"foggy_{image_path.stem}_vis_{args.visibility}.png"
                output_path = os.path.join(camera_output_dir, foggy_file_name)

                foggy_image = Image.fromarray(foggy_image_np)
                foggy_image.save(output_path)
            
            except Exception as e:
                logging.error(f"ERROR processing {image_path.name}: {e}")

            print("Camera simulation complete.\n")

    if include_lidar:
        lidar_paths = list(args.input_lidar_dir.glob('*.npy'))
        
        for lidar_path in tqdm(lidar_paths, desc="LiDAR Fog Simulation", leave=False):
            try:
                # Load point cloud from the file and simulate fog
                point_cloud_np = np.load(lidar_path)
                foggy_pc_np = lidar_fog.simulate(point_cloud_np, visibility=args.visibility, max_range_for_vis=250)

                # Save the foggy point cloud
                foggy_file_name = f"foggy_{lidar_path.stem}_vis_{args.visibility}.npy"
                output_path = os.path.join(lidar_output_dir, foggy_file_name)
                np.save(output_path, foggy_pc_np)

            except Exception as e:
                logging.error(f"ERROR processing {lidar_path.name}: {e}")

            print("LiDAR simulation complete.")
