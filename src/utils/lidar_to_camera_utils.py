# The methods in LiDARToCamera are adapted from the Waymo Open Dataset tutorial:
# https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial.ipynb
# 
# The original code is licensed under the Apache License 2.0. 
# See LICENSES/WAYMO_LICENSE for details.


import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .lidar_utils import point_cloud_conversion


class LidarToCamera:
    def __init__(self, frame, frame_num, folder_paths, camera_id=1):
        """
        Initialise a class that projects 3D LiDAR points onto a 2D camera.

        Camera name IDs from waymo_open_dataset/dataset_pb2:
        1 - FRONT
        2 - FRONT_LEFT
        3 - FRONT_RIGHT
        4 - SIDE_LEFT
        5 - SIDE_RIGHT

        Args:
            frame (dataset_pb2.Frame):
                The read frame from the tfrecord file.
            frame_num (int):
                The frame number within the tfrecord scene.
            folder_paths (dict):
                A dictionary containing the paths of the tfrecord file output and the corresponding sensor subfolders.
                Dictionary key contains: 
            camera_id (int):
                The ID of the desired camera to show the projection points (default: 1).
        """
        self.frame = frame
        self.frame_num = frame_num
        self.camera_id = camera_id
        self.main_folder_dir = folder_paths["main_folder"]

        camera_folder_map = {
            1: folder_paths["cam_front"],
            2: folder_paths["cam_front_left"],
            3: folder_paths["cam_front_right"],
            4: folder_paths["cam_side_left"],
            5: folder_paths["cam_side_right"]
        }

        self.root_dir = camera_folder_map.get(camera_id)    
        if self.root_dir is None:
            raise ValueError(f"Invalid camera ID: {camera_id}")  

        # Select the image corresponding to the target camera 
        for image in self.frame.images:
            if image.name == self.camera_id:
                self.frame_image = image   
            
    def get_projected_points(self):
        """
        Gets the camera projection points for the specified camera, which includes the projection coordinates and depth.

        Returns:
            projected_points (np.ndarray):
                The projected points of size (N, 3) for the specified camera that contains the projection coordinates (x, y) and the depth of the LiDAR points.
        """
        points_all_ri1, _, cp_points_all_ri1, _ = point_cloud_conversion(self.frame)

        # Calculate the distance of the 3D LiDAR points from the vehicle
        points_all_ri1_tensor = tf.norm(points_all_ri1, axis=-1, keepdims=True)
        cp_points_all_ri1_tensor = tf.constant(cp_points_all_ri1, dtype=tf.int32)

        # Create a mask that selects points whose camera ID matches the desired camera
        mask = tf.equal(cp_points_all_ri1_tensor[..., 0], self.camera_id)

        # Pick the rows where the mask was True and change to float for the camera projections
        cp_points_all_ri1_tensor = tf.cast(tf.gather_nd(cp_points_all_ri1_tensor, tf.where(mask)), dtype=tf.float32)
        points_all_ri1_tensor = tf.gather_nd(points_all_ri1_tensor, tf.where(mask))

        # Concatenate the projection coordinates with its corresponding depth
        projected_points = tf.concat([cp_points_all_ri1_tensor[..., 1:3], points_all_ri1_tensor], axis=-1).numpy()

        return projected_points

    def get_proj_coords_and_depth(self):
        """
        Extracts each of the x-coordinates, y-coordinates and depth values of the projected LiDAR points in the imge.

        Returns:
            self.xs (list):
                The x-coordinates of the projected LiDAR points in the image.
            self.ys (list):
                The y-coordinates of the projected LiDAR points in the image.
            self.colours (list):
                The depth-based colour values
            self.sparse_depth_points (np.ndarray):
                The points for a 2D sparse depth map with the same shape as the input image
        """
        projected_points = self.get_projected_points()

        self.xs = []
        self.ys = []
        depth = []
        self.colours = []

        # Seperate the x-coordinates, y-coordinates and the depth
        for point in projected_points:
            self.xs.append(point[0])
            self.ys.append(point[1])
            depth.append(point[2])

            # Generate a colour based on the depth of the LiDAR point with relative distance of 20m - for visualisation only
            c = plt.get_cmap('jet')((point[2] % 20) / 20)
            c = list(c)
            c[-1] = 0.5

            self.colours.append(c)

        # Create the sparse depth map array and initialise it with zeros
        image_height = tf.image.decode_jpeg(self.frame_image.image).shape[0]
        image_width = tf.image.decode_jpeg(self.frame_image.image).shape[1]
        init_depth_points = np.zeros((image_height, image_width), dtype=np.float32)

        # Populate the array
        for i in range(len(self.xs)):
            x = int(self.xs[i])
            y = int(self.ys[i])
            d = depth[i]

            # Check if the point is within the image bounds
            if 0 <= y < image_height and 0 <= x < image_width:
                init_depth_points[y, x] = d

        save_path = os.path.join(self.root_dir, "sparse_depth_maps")
        os.makedirs(save_path, exist_ok=True)

        # Multiply by a large number to visualise the points clearly
        self.sparse_depth_points = (init_depth_points * 5000).astype(np.uint16)
        # self.sparse_depth_points = init_depth_points.astype(np.uint16)

        return self.xs, self.ys, self.colours, self.sparse_depth_points
    
    def save_projected_points_map(self):
        """
        Plot the projection points on the camera image and save it.
        """
        plt.figure(figsize=(20, 12))
        plt.imshow(tf.image.decode_jpeg(self.frame_image.image))
        plt.grid("off")
        plt.axis("off")
        plt.scatter(self.xs, self.ys, c=self.colours, s=5.0, edgecolors="none")

        save_path = os.path.join(self.root_dir, "projection_maps")
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"frame_{self.frame_num}.png"), bbox_inches='tight')
        plt.close()

    def save_sparse_depth_map(self, png_image=False):
        """
        Create a sparse depth map from the sparse depth points and save it.

        Args:
            png_image (bool):
                Whether to save the sparse depth map in its original form as a numpy array or as a png image (default: False).
        """
        """
        save_path = os.path.join(self.root_dir, "sparse_depth_maps")
        os.makedirs(save_path, exist_ok=True)

        if png_image:
            # Scale depth values to milimetres to easily visualise the points
            self.sparse_depth_points = (self.sparse_depth_points * 1000).astype(np.uint16)

            img = Image.fromarray(self.sparse_depth_points, mode='I;16')
            img.save(os.path.join(save_path, f"frame_{self.frame_num}.png"))
        else:
            np.save(os.path.join(save_path, f"frame_{self.frame_num}.npy"), self.sparse_depth_points)
        """
        path = os.path.basename(self.main_folder_dir)
        path = path.split("_")[0]
        save_path = os.path.join("/home/administrator/Clear2Fog/output_/depth_map", f"frame_{path}.png")
        img = Image.fromarray(self.sparse_depth_points, mode='I;16')
        img.save(save_path)
