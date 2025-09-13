import os
import random
import math
import numpy as np
from PIL import Image
from skimage.morphology import erosion, footprint_rectangle
from skimage.color import rgb2gray
from skimage.util import img_as_float


class FogCamera:
    def __init__(self, depth_obj, input_file_path, min_depth=1000):
        """
        Initialises FogCamera and computes the average atmospheric light from input images.

        If no valid atmospheric light can be estimated, a default value of [0.8, 0.8, 0.8] is used.

        Args:
            depth_obj: 
                Object of class DepthGenerator that returns a metric depth map.
            input_file_path (str): 
                Path to the folder containing the input RGB images. Images are expected to be in RGB format, shape (H, W, 3), dtype=uint8, values in [0, 255].
            min_depth (float): 
                Minimum depth to consider for atmospheric light selection (default: 1000).
        """
        # Define the bounds for the atmospheric light
        self.lower_bound_intensity = 162 / 255.0
        self.upper_bound_intensity = 220 / 255.0

        # Calculate an average atmospheric light value for the given dataset
        all_files = os.listdir(input_file_path)
        if len(all_files) > 100:
            selected_files = random.sample(all_files, 100)
        else:
            selected_files = all_files

        atmospheric_lights = []
        for image_file in selected_files:
            image_path = os.path.join(input_file_path, image_file)
            image = Image.open(image_path).convert("RGB")
            image_original_np = np.array(image)

            depth_pred_np = depth_obj.run(image_path)

            atmospheric_light = self.estimate_atmospheric_light(image_original_np, depth_pred_np, min_depth=min_depth)

            if atmospheric_light is not None:
                atmospheric_lights.append(atmospheric_light)

        # Calculate the average atmospheric light across the selected images
        if len(atmospheric_lights) > 0:
            self.avg_atmospheric_light = np.mean(atmospheric_lights, axis=0)
        else:
            self.avg_atmospheric_light = np.array([0.8, 0.8, 0.8], dtype=np.float32)  # Fallback

    def transmittance(self, visibility, depth_pred_np):
        """
        Calculate the transmittance map t(x, y).

        Args:
            visibility (float): 
                The desired meteorological optical range (MOR) in metres.
            depth_pred_np (np.ndarray): 
                Depth map, shape (H, W), dtype=float32, values in metres.

        Returns:
            t (np.ndarray): 
                Transmittance map of shape (H, W), dtype=float32, values in [0, 1].
        """
        beta= -np.log(0.05) / visibility
        t = np.exp(-beta * depth_pred_np)
        
        return t
    
    def estimate_atmospheric_light(self, input_np, depth_map_np, min_depth=1000):
        """
        Estimate atmospheric light (L) in an image using the dark channel prior.

        Args:
            input_np (np.ndarray): 
                Input RGB image, shape (H, W, 3), dtype=uint8, values in [0, 255].
            depth_map_np (np.ndarray): 
                Depth map, shape (H, W), dtype=float32, values in metres.
            min_depth (float): 
                Minimum depth to consider for atmospheric light selection (default: 1000).

        Returns:
            atmospheric_light (np.ndarray | None):
                Estimated atmospheric light, shape (3,), dtype=float32, values in [0, 1]. Returns None if no valid candidates are found.
        """
        patch_size = 15
        footprint = footprint_rectangle((patch_size, patch_size))

        float_image = img_as_float(input_np)
        
        # Apply morphological erosion to each channel
        # Dark channel is the pixel-wise minimum of eroded R, G and B
        eroded_r = erosion(float_image[:, :, 0], footprint)
        eroded_g = erosion(float_image[:, :, 1], footprint)
        eroded_b = erosion(float_image[:, :, 2], footprint)
        dark_channel = np.minimum(np.minimum(eroded_r, eroded_g), eroded_b)

        # Take top 0.1% of brightest pixels in dark channel
        height, width, _ = float_image.shape
        num_pixels = height * width
        brightest_pixels_count = max(1, math.ceil(num_pixels / 1000))

        # Flatten dark channel and sort indices in descending order
        flat_dark_channel = dark_channel.flatten()
        indices = np.argsort(flat_dark_channel)[::-1]
        
        # Keep only indices where depth >= min_depth
        if depth_map_np is not None:
            flat_depth = depth_map_np.flatten()
            indices = indices[flat_depth[indices] >= min_depth]

        # If no candidates remain, return None
        if len(indices) == 0:
            return None

        # Take the top brightest pixels
        brightest_pixels_indices = indices[:brightest_pixels_count]

        # Find the median intensity amongst the grayscale intensities of the brightest candidates
        flat_i_gray = rgb2gray(float_image).flatten()
        i_gray_brightest = flat_i_gray[brightest_pixels_indices]

        # Find which candidate corresponds to the median intensity
        median_intensity = np.median(i_gray_brightest)
        closest_index = np.argmin(np.abs(i_gray_brightest - median_intensity))
        index_of_median_in_brightest = closest_index

        # Get global index of this pixel
        index_l = brightest_pixels_indices[index_of_median_in_brightest]
        
        # Estimate the atmospheric light at the chosen pixel
        row_l, col_l = np.unravel_index(index_l, (height, width))
        atmospheric_light = float_image[row_l, col_l, :]

        return atmospheric_light

    def simulate(self, original_np, input_np, depth_map_np, visibility, min_depth=1000):
        """
        Simulate fog on a clear input image using Koschmieder's law.

        Args:
            original_np (np.ndarray):
                Original RGB image, shape (H, W, 3), dtype=uint8, values in [0, 255].
            input_np (np.ndarray):
                Input RGB image to be fogged, shape (H, W, 3), dtype=uint8, values in [0, 255].
            depth_map_np (np.ndarray): 
                Depth map, shape (H, W), dtype=float32, values in metres.
            visibility (float): 
                The desired meteorological optical range (MOR) in metres.
            min_depth (float): 
                Minimum depth to consider for atmospheric light selection (default: 1000).

        Returns:
            I_foggy (np.ndarray): 
                Foggy RGB image, shape (H, W, 3), dtype=uint8, values in [0, 255].
        """
        I_clear = input_np.astype(np.float32) / 255.0

        L = self.estimate_atmospheric_light(original_np, depth_map_np, min_depth=min_depth)
        
        # For the cases where the sky is not visible, use the average calculated when initialising the class
        if L is None:
            L = self.avg_atmospheric_light

        # Clip the luminance to the desired range
        luminance = 0.2126 * L[0] + 0.7152 * L[1] + 0.0722 * L[2]
        clipped_luminance = np.clip(luminance, self.lower_bound_intensity, self.upper_bound_intensity)
        L = np.array([clipped_luminance, clipped_luminance, clipped_luminance], dtype=np.float32)

        t = self.transmittance(visibility, depth_map_np)

        # Use Koschmieder's law to generate the foggy image
        I_foggy = I_clear * t[:, :, np.newaxis] + L[np.newaxis, np.newaxis, :] * (1 - t[:, :, np.newaxis])
        I_foggy = (np.clip(I_foggy * 255.0, 0, 255)).astype(np.uint8)

        return I_foggy
