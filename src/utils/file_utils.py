import os
import tensorflow as tf
from tqdm import tqdm

from src.waymo_protos import dataset_pb2 as open_dataset


def read_waymo_frames(tf_file_path, n_frames=1):
    """
    Read the frames from the tfrecord file.

    Adapted from the Waymo Open Dataset tutorial:
        https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial.ipynb
        The original code is licensed under the Apache License 2.0. 
        See LICENSES/WAYMO_LICENSE for details.

    Args:
        tf_file_path (str): 
            The file path to the tfrecord.
        n_frames (int): 
            The number of frames that need to be extracted from the tfrecord file - set to a very high number if all frames are required (default: 1).
            

    Returns:
        frames (list of open_dataset.Frame): 
            A list containing the extracted frames.
    """
    dataset = tf.data.TFRecordDataset(tf_file_path, compression_type='')

    frames = []
    i = 0
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytes(data.numpy()))

        frames.append(frame)

        i += 1
        if i == n_frames:
            break

    return frames


def delete_waymo_files(main_folder):
    """
    Keep only the "Day" and "sunny" images from the Waymo dataset to simulate fog on and delete the rest.

    Args:
        main_folder (str):
            The main Waymo folder contining the training, test and validation folders under it.
    """
    for folder in os.listdir(main_folder):
        folder_path = os.path.join(main_folder, folder)

        removed_days = 0
        for file in tqdm(os.listdir(folder_path), desc="Deleting unwanted files", leave=False):
            file_path = os.path.join(folder_path, file)

            frames = read_waymo_frames(file_path, n_frames=1)
            for frame in frames:
                if frame.context.stats.time_of_day != "Day" or frame.context.stats.weather != "sunny":
                        os.remove(file_path)
                        removed_days += 1

        break
    
    print(f"Total removed days: {removed_days}")


def output_folders_waymo(tf_file, output_dir):
    """
    Structure the output directory such that the tfrecord outputs are organised into numbered subfolder each containing a maximum of 20 tfrecords.
    
    Each tfrecord folder will contain subfolders for camera and LiDAR foggy data.

    Args:
        tf_file (str):
            The tfrecord file name.
        output_dir (str):
            The root output directory where fog-simulated folders will be stored.
    
    Returns:
        folder_paths (dict):
            A dictionary mapping the main folder and its corresponding sensor folders to their corresponding paths.
    """
    subfolders = []
    subfolders_sorted = []

    for folder in os.listdir(output_dir):
        if os.path.isdir(os.path.join(output_dir, folder)):
            subfolders.append(folder)

    # If output directory is empty, create the first folder
    if len(subfolders) == 0:
        folder = os.path.join(output_dir, "1")
        os.mkdir(folder)
    else:
        for name in subfolders:
            subfolders_sorted.append(int(name))
        
        last_folder_num = sorted(subfolders_sorted)[-1]
        last_folder_path = os.path.join(output_dir, str(last_folder_num))

        # If the numbered folder contains more than 20 folders, place the new tfrecord folder in a new numbered folder
        num_folders = len(os.listdir(last_folder_path))
        if num_folders < 20:
            folder = last_folder_path
        else:
            new_folder_num = last_folder_num + 1
            folder = os.path.join(output_dir, str(new_folder_num))
            os.mkdir(folder)

    tf_folder_name = tf_file.split("_")[0]
    tf_folder_path = os.path.join(folder, tf_folder_name)
    os.makedirs(tf_folder_path, exist_ok=True)

    # In each tfrecord folder, create the following subfolders to store the relevant foggy data
    folder_names = ["cam_front", "cam_front_left", "cam_front_right", "cam_side_left", "cam_side_right", "lidar"]
    folder_paths = {}
    for name in folder_names:
        path = os.path.join(tf_folder_path, name)
        os.makedirs(path, exist_ok=True)
        folder_paths[name] = path

    folder_paths["main_folder"] = tf_folder_path

    return folder_paths


def organise_output(output_dir, camera=True, lidar=False):
    """
    Create the top-level output directories for storing the fog-simulated data.

    Args:
        output_dir (str):
            The root directory where the fog-simulated data folders will be created.
        camera (bool, optional):
            Whether to create the camera output folder (default: True).
        lidar (bool, optional):
            Whether to create the LiDAR output folder (default: False).

    Returns:
        camera_output_dir (str):
            Path to the output camera foggy folder.
        lidar_output_dir (str):
            Path to the output LiDAR foggy folder.
    """
    camera_output_dir = None
    lidar_output_dir = None

    if camera:
        camera_output_dir = os.path.join(output_dir, "foggy_camera")
        os.makedirs(camera_output_dir, exist_ok=True)

    if lidar:
        lidar_output_dir = os.path.join(output_dir, "foggy_lidar")
        os.makedirs(lidar_output_dir, exist_ok=True)

    return camera_output_dir, lidar_output_dir
