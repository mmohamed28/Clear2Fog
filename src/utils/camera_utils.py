# The methods in Camera are adapted from the Waymo Open Dataset tutorial:
# https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial.ipynb
# 
# The original code is licensed under the Apache License 2.0. 
# See LICENSES/WAYMO_LICENSE for details.


import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


class Camera:
    def __init__(self, frame, frame_num, folder_paths, camera_id=1):
        """
        Initialise a class to utilise the camera sensors.

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
                self.camera_labels = frame.camera_labels

    def decode_camera_image(self, labels=False):
        """
        Decode a camera image from a frame and include its corresponding bounding box labels if required.

        Args:
            labels (bool):
                Whether to show the bounding box labels on the image (default: False).

        Returns:
            self.final_image (np.ndarray):
                The decoded image as a numpy array
        """
        # Decode the image and convert to a NumPy array
        image_np = tf.image.decode_jpeg(self.frame_image.image).numpy()
        
        image_pil = Image.fromarray(image_np)
        draw = ImageDraw.Draw(image_pil)

        if labels:
            for cam_label in self.camera_labels:
            # Draw the bounding boxes only if the labels come from the same camera as the image (i.e. camera name for the labels = camera name for the image)
                if cam_label.name == self.frame_image.name:  
                    for label in cam_label.labels:
                        x_min = label.box.center_x - 0.5 * label.box.length
                        y_min = label.box.center_y - 0.5 * label.box.width
                        x_max = label.box.center_x + 0.5 * label.box.length
                        y_max = label.box.center_y + 0.5 * label.box.width
                        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=2) 
    
        # Convert back to numpy
        self.final_image = np.array(image_pil)

        return self.final_image

    def save_camera_image(self):
        """
        Save the camera image as a PNG file.
        """
        img_path = os.path.join(self.root_dir, f"frame_{self.frame_num}.png")
        plt.imsave(img_path, self.final_image)
        