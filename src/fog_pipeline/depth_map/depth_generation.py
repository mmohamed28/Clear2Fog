# The class DepthGenerator is adapted from Depth Pro:
# https://github.com/apple/ml-depth-pro
# 
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
# See LICENSES/DEPTH_PRO for the full license text.


import torch

from .depth_pro import create_model_and_transforms
from .depth_pro.utils import load_rgb


# Download the Depth Pro checkpoint:
#   wget https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt
# Place it in: ./depth_pro/checkpoints
class DepthGenerator:
    def __init__(self, device="cuda", precision=torch.float32):
        """
        Initialise a DepthGenerator instance by loading the Depth Pro model and its preprocessing transform.

        Args:
            device (str):
                Device to run the model on: "cuda", "cpu" or "mps" (default: "cuda").
            precision (torch.dtype):
                Torch dtype for model inference (default: torch.float32)
        """
        self.precision = precision
        
        if device == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Load model and preprocessing transform
        self.model, self.transform = create_model_and_transforms(
            device=self.device,
            precision=self.precision
        )
        self.model.eval()

    def run(self, input_img_path):
        """
        Generate a metric depth map for a single RGB image using the Depth Pro model.

        Args:
            input_img_path (str):
                Path to an input RGB image file. Images are expected to be in RGB format, shape (H, W, 3), dtype=uint8, values in [0, 255].

        Returns:
            metric_depth_pred (np.ndarray):
                A 2D NumPy array of shape (H, W) where each element is the estimated depth in metres for the corresponding pixel.
        """
        # Load and preprocess an image
        image, _, f_px = load_rgb(input_img_path)
        image = self.transform(image)

        # Run inference and return the prediction as a NumPy array
        prediction = self.model.infer(image, f_px=f_px)
        metric_depth_pred = prediction["depth"].cpu().detach().numpy()

        return metric_depth_pred
