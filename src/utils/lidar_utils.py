import os
import numpy as np
import open3d as o3d
import tensorflow as tf

from . import frame_utils, transform_utils


def point_cloud_conversion(frame):
        """
        Convert the range images to point cloud.

        Adapted from the Waymo Open Dataset tutorial:
            https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial.ipynb
            The original code is licensed under the Apache License 2.0. 
            See LICENSES/WAYMO_LICENSE for details.

        Args:
            frame (dataset_pb2.Frame):
                The read frame from the tfrecord file.

        Returns:
            points_all_ri1 (np.ndarray):
                The concatenated point clouds from the first LiDAR return.
            points_all_ri2 (np.ndarray):
                The concatenated point clouds from the second LiDAR return.
            cp_points_all_ri1 (np.ndarray):
                The corresponding camera projection points for the first LiDAR return.
            cp_points_all_ri2 (np.ndarray):
                The corresponding camera projection points for the second LiDAR return.
        """
        # Parse the range images, camera projections and range image pose for the TOP LiDAR
        range_images, camera_projections, _, range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(frame)

        # Convert the range images to point clouds
        points_ri1, cp_points_ri1 = frame_utils.convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose, ri_index=0)
        points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose, ri_index=1)

        # Concatenate the point clouds and camera projections from the different LiDAR sensors into individual numpy arrays
        points_all_ri1 = np.concatenate(points_ri1, axis=0)
        points_all_ri2 = np.concatenate(points_ri2, axis=0)
        cp_points_all_ri1 = np.concatenate(cp_points_ri1, axis=0)
        cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)

        return points_all_ri1, points_all_ri2, cp_points_all_ri1, cp_points_all_ri2


def display_point_cloud(point_cloud, bboxes=None):
    """
    Display the point cloud with its corresponding 3D bounding boxes.

    Adapted from: 
        https://github.com/aabramovrepo/python-projects-blog/blob/main/waymo_dataset/visualization/visu_point_cloud.py
        The original code is licensed under MIT License
        See LICENSES/ALEXEY_MIT_LICENSE for full license text.

    Args:
        point_cloud (np.ndarray):
            A NumPy array of shape (N, 3) representing 3D point cloud coordinates.
        bboxes (list[np.ndarray], optional):
            A list of bounding boxes where each bounding box is a NumPy array of shape (8, 3) representing the 8 corners of a 3D box.
    """
    line_segments = [[0, 1], [1, 3], [3, 2], [2, 0],
                    [4, 5], [5, 7], [7, 6], [6, 4],
                    [0, 4], [1, 5], [2, 6], [3, 7]]

    # Create a window for 3D visualisation
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    # Set the viewer appearance with a black background and point cloud size of 2
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.0

    # Load the point cloud points and add a coordinate axis
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.6, origin=[0, 0, 0])

    # Add the point cloud and the coordinate axis to the viewer
    vis.add_geometry(pcd)
    vis.add_geometry(mesh_frame)

    if bboxes:
        for bbox_3d_points in bboxes:
            colors = [[1, 0, 0] for _ in range(len(line_segments))]
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(bbox_3d_points),
                lines=o3d.utility.Vector2iVector(line_segments),
            )

            # Add the box to the viewer
            line_set.colors = o3d.utility.Vector3dVector(colors)
            vis.add_geometry(line_set)

    vis.run()


class Lidar:
    def __init__(self, frame, frame_num, folder_paths):
        """
        Initialise a class to utilise the LiDAR sensor.

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
        self.main_folder_dir = folder_paths["main_folder"]
        self.lidar_dir = folder_paths["lidar"]
    
    def build_3d_bbox(self, label):
        """
        Build the full 3D bounding box for the given label.

        Adapted from: 
            https://github.com/aabramovrepo/python-projects-blog/blob/main/waymo_dataset/visualization/visu_point_cloud.py
            The original code is licensed under MIT License
            See LICENSES/ALEXEY_MIT_LICENSE for full license text.
        
        Args:
            label (dataset_pb2.Label):
                The object label from the given frame.

        Returns:
            bbox_3d_points (np.ndarray):
                The coordinates of the 3D bounding box for the given label.
        """

        # Get the dimensions for the width, length and heading of the given label
        # Heading is reversed as the system used in Waymo is reversed compared to the visualiser
        width, length, heading = label.box.width, label.box.length, -label.box.heading

        # Get the dimensions for the centre of the label
        x, y, z = label.box.center_x, label.box.center_y, label.box.center_z

        # Construct the 2D bottom face of the 3D bounding box centred around (0, 0)
        bbox_2d_corners = np.array([[-0.5 * length, -0.5 * width],
                                    [-0.5 * length, 0.5 * width],
                                    [0.5 * length, -0.5 * width],
                                    [0.5 * length, 0.5 * width]])
        
        # This function provides the 3D rotation matrix for the given heading
        mat = transform_utils.get_yaw_rotation(heading)
        # Extract only the the top left 2x2 matrix as the 2D box is getting rotated in the x-y plane
        rot_mat = mat.numpy()[:2, :2]

        transformed_bbox = bbox_2d_corners @ rot_mat

        # Calculate how far the label stretches vertically
        z_bottom = z - label.box.height / 2
        z_top = z + label.box.height / 2

        # Initialise a placeholder for the 3D bounding box points - creates an array of 8 points
        bbox_3d_points = [[0.0, 0.0, 0.0]] * transformed_bbox.shape[0] * 2

        for idx in range(transformed_bbox.shape[0]):
            # Shift the x and y coordinate of the centre of the label to the transformed values
            x_, y_ = x + transformed_bbox[idx][0], y + transformed_bbox[idx][1]

            # Stretch the points vertically to create the 3D box
            bbox_3d_points[idx] = [x_, y_, z_bottom]
            bbox_3d_points[idx + 4] = [x_, y_, z_top]

        return bbox_3d_points
    
    def get_pc_and_bboxes(self, include_all_ri1=True, include_all_ri2=True):
        """
        Extract the point cloud points and corresponding 3D bounding boxes data.

        Args:
            include_all_ri1 (bool):
                Whether to include the point clouds from the first LiDAR return in the visualiser (default: True).
            include_all_ri2 (bool):
                Whether to include the point clouds from the second LiDAR return in the visualiser (default: True).

        Returns:
            self.point_cloud (np.ndarray):
                The point cloud points in metres for the given frame.
            self.bboxes (list of np.ndarray):
                The 3D bounding box points for the given frame.
        """
        # Load point cloud
        points_all_ri1, points_all_ri2, _, _ = point_cloud_conversion(self.frame)

        if include_all_ri1 is True and include_all_ri2 is False:
            self.point_cloud = points_all_ri1

        elif include_all_ri1 is False and include_all_ri2 is True:
            self.point_cloud = points_all_ri2

        else:
            self.point_cloud = np.concatenate([points_all_ri1, points_all_ri2], axis=0)

        self.bboxes = []
        for label in self.frame.laser_labels:
            # Get the 3D bounding box for the given label
            bbox_3d_points = self.build_3d_bbox(label)

            self.bboxes.append(bbox_3d_points)

        return self.point_cloud, self.bboxes
    
    def get_pc_intensity_and_bboxes(self, include_all_ri1=True, include_all_ri2=True):
        """
        Extract the point cloud points and corresponding intensity values along with the 3D bounding boxes data.

        Args:
            include_all_ri1 (bool):
                Whether to include the point clouds from the first LiDAR return in the visualiser (default: True).
            include_all_ri2 (bool):
                Whether to include the point clouds from the second LiDAR return in the visualiser (default: True).

        Returns:
            self.point_cloud_intensity (np.ndarray):
                The point cloud points in metres along with corresponding intensity value with shape (N, 4) containing [x, y, z, intensity], where intensity is in range [0, 255].
            self.bboxes (list of np.ndarray):
                The 3D bounding box points for the given frame.
        """
        # Parse the range images
        range_images, _, _, _ = frame_utils.parse_range_image_and_camera_projection(self.frame)
        
        # Load point cloud
        points_all_ri1, points_all_ri2, _, _ = point_cloud_conversion(self.frame)

        intensity_all_ri1 = []
        intensity_all_ri2 = []

        self.frame.lasers.sort(key=lambda laser: laser.name)
        for laser in self.frame.lasers:
            # Process Return 1
            ri_1 = range_images[laser.name][0]
            ri_1_tensor = tf.convert_to_tensor(ri_1.data)
            ri_1_tensor = tf.reshape(ri_1_tensor, ri_1.shape.dims)

            mask_ri_1 = ri_1_tensor[..., 0] > 0

            intensity_1_tensor = ri_1_tensor[..., 1]
            valid_intensity_ri1 = tf.boolean_mask(intensity_1_tensor, mask_ri_1)
            intensity_all_ri1.append(valid_intensity_ri1.numpy())

            # Process Return 2
            ri_2 = range_images[laser.name][1]
            ri_2_tensor = tf.convert_to_tensor(ri_2.data)
            ri_2_tensor = tf.reshape(ri_2_tensor, ri_2.shape.dims)

            mask_ri_2 = ri_2_tensor[..., 0] > 0

            intensity_2_tensor = ri_2_tensor[..., 1]
            valid_intensity_ri2 = tf.boolean_mask(intensity_2_tensor, mask_ri_2)
            intensity_all_ri2.append(valid_intensity_ri2.numpy())

        # Concatenate intensities from all sensors for each return
        intensity_all_ri1 = np.concatenate(intensity_all_ri1, axis=0)
        intensity_all_ri2 = np.concatenate(intensity_all_ri2, axis=0)

        if include_all_ri1 is True and include_all_ri2 is False:
            points_with_intensity_ri1 = np.c_[points_all_ri1, intensity_all_ri1]
            self.point_cloud = points_with_intensity_ri1

        elif include_all_ri1 is False and include_all_ri2 is True:
            points_with_intensity_ri2 = np.c_[points_all_ri2, intensity_all_ri2]
            self.point_cloud = points_with_intensity_ri2

        else:
            points_with_intensity_ri1 = np.c_[points_all_ri1, intensity_all_ri1]
            points_with_intensity_ri2 = np.c_[points_all_ri2, intensity_all_ri2]
            self.point_cloud = np.concatenate([points_with_intensity_ri1, points_with_intensity_ri2], axis=0)

        # Scale the final intensity column (index 3) from [0, 1] to [0, 255]
        self.point_cloud[:, 3] = (self.point_cloud[:, 3] * 255).astype(np.uint8)

        self.bboxes = []
        for label in self.frame.laser_labels:
            # Get the 3D bounding box for the given label
            bbox_3d_points = self.build_3d_bbox(label)

            self.bboxes.append(bbox_3d_points)

        return self.point_cloud, self.bboxes

    def save_point_cloud_and_3d_bbox(self):
        """
        Save the point cloud and 3D bounding boxes to .npy files.

        Args:
            include_all_ri1 (bool):
                Whether to include the point clouds from the first LiDAR return in the visualiser (default: True).
            include_all_ri2 (bool):
                Whether to include the point clouds from the second LiDAR return in the visualiser (default: True).
        """
        # Save the point cloud
        save_path = os.path.join(self.lidar_dir, "point_clouds")
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, f'frame_{self.frame_num}.npy'), self.point_cloud)

        # Save the 3D bounding boxes
        save_path = os.path.join(self.lidar_dir, "3d_bboxes")
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, f'frame_{self.frame_num}.npy'), self.bboxes)
