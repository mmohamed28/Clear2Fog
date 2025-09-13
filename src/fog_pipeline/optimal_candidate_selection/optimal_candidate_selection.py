import os
import cv2
import multiprocessing
import torch
import numpy as np
from brisque.brisque import BRISQUE
from torchvision import ops
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

from .AuthESI import authenticity


brisque_obj = BRISQUE(url=False)


def evaluate_image(image):
    """
    Evaluates an image using both AuthESI and BRISQUE metrics.

    Args:
        image (np.ndarray): 
            Input image in BGR format, shape (H, W, 3), dtype=uint8, values [0, 255].

    Returns:
        dict: Dictionary with the keys:
            - "auth_esi" (float): AuthESI score, lower is better..
            - "brisque" (float): BRISQUE score, lower is better..
    """
    auth_esi_score = authenticity(image)
    brisque_score = brisque_obj.score(image)

    return {'auth_esi': auth_esi_score, 'brisque': brisque_score}


class OCS:
    def __init__(self, device="cuda", box_thresh=0.5, iou_thresh=0.5):
        """
        Initialises the Optimal Input Selection (OCS) pipeline.

        Args:
            device (str): 
                Device to run the detection model on: "cuda", "cpu", or "mps" (default: "cuda").
            box_thresh (float): 
                Confidence threshold for object detection boxes.
            iou_thresh (float): 
                IoU threshold for counting retained objects.
        """
        self.box_thresh = box_thresh
        self.iou_thresh = iou_thresh

        if device == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = fasterrcnn_resnet50_fpn(pretrained=True).to(self.device)
        self.model.eval()

    def detect_boxes(self, image_np):
        """
        Detects objects in an image using Faster R-CNN.

        Args:
            image_np (np.ndarray): 
                Input RGB image as a NumPy array, shape (H, W, 3), dtype=uint8, values in [0, 255].

        Returns:
            boxes_keep (np.ndarray): 
                Detected bounding boxes with shape (N, 4), where N is the number of boxes.
        """
        img_tensor = F.to_tensor(image_np).to(self.device)

        with torch.no_grad():
            output = self.model([img_tensor])[0]

        boxes = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()

        keep = scores >= self.box_thresh
        boxes_keep = boxes[keep]

        return boxes_keep

    def evaluate_object_retention(self, foggy_boxes, reference_boxes):
        """
        Computes object retention score by comparing foggy boxes with reference boxes.

        Score is structured so that 0 = all objects retained (best), 1 = no objects retained (worst).

        Args:
            foggy_boxes (np.ndarray): 
                Boxes detected in the foggy image.
            reference_boxes (np.ndarray): 
                Boxes detected in the reference image.

        Returns:
            object_retention_score (float | None): 
                Object retention score, or None if reference_boxes is empty.
        """
        if len(reference_boxes) == 0:
            return None

        if len(foggy_boxes) == 0:
            return 1.0

        ref_boxes_tensor = torch.from_numpy(np.array(reference_boxes)).float().to(self.device)
        foggy_boxes_tensor = torch.from_numpy(np.array(foggy_boxes)).float().to(self.device)
        iou_matrix = ops.box_iou(ref_boxes_tensor, foggy_boxes_tensor)

        max_iou_per_ref_box, _ = iou_matrix.max(dim=1)
        retained_boxes_count = (max_iou_per_ref_box >= self.iou_thresh).sum().item()

        # Minimise score
        object_retention_score = 1.0 - (retained_boxes_count / len(reference_boxes))
        
        return object_retention_score

    def metric_evaluation_scores(self, foggy_images):
        """
        Computes AuthESI and BRISQUE scores for multiple images in parallel.

        Args:
            foggy_images (list[np.ndarray]): 
                List of foggy images in BGR format, shape (H, W, 3), dtype=uint8, values in [0, 255]

        Returns:
            all_scores (list[dict]): 
                List of dictionaries containing 'auth_esi' and 'brisque' for each image.
        """
        num_images = len(foggy_images)

        with multiprocessing.Pool(processes=min(num_images, os.cpu_count())) as pool:
            all_scores = pool.map(evaluate_image, foggy_images)
            
        return all_scores
    
    def find_best_result(self, all_scores):
        """
        Computes weighted combination of object retention, AuthESI, and BRISQUE to select best image.

        Lower combined score = better image.

        Args:
            all_scores (list[dict]): 
                List of score dictionaries for each candidate image.

        Returns:
            best_index (int): 
                Index of the best image in the candidate list.
            best_raw_scores (dict): 
                Raw metrics for the best image.
        """
        # Extract raw scores
        auth_esi_scores = np.array([score['auth_esi'] for score in all_scores])
        brisque_scores = np.array([score['brisque'] for score in all_scores])

        # Normalise AuthESI and BRISQUE to [0, 1]
        norm_auth = (auth_esi_scores - auth_esi_scores.min()) / (auth_esi_scores.max() - auth_esi_scores.min() + 1e-6)
        norm_brisque = (brisque_scores - brisque_scores.min()) / (brisque_scores.max() - brisque_scores.min() + 1e-6)

        final_scores = []
        for i, score in enumerate(all_scores):
            # Set weights
            w_auth, w_retention, w_brisque = 0.5, 0.3, 0.2

            # If no object is retained, distribute its weight amongst the other two weights
            if score['object_retention'] is None:
                w_auth += w_retention / 2
                w_brisque += w_retention / 2
                retention_val = 0.0
            else:
                retention_val = score['object_retention']

            # Weighted sum
            final_score = (
                w_auth * norm_auth[i] +
                w_brisque * norm_brisque[i] +
                w_retention * retention_val
            )
            final_scores.append(final_score)

        # Select image with the lowest weighted score
        final_scores = np.array(final_scores)
        best_index = np.argmin(final_scores)

        score_default = final_scores[0]
        score_best = final_scores[best_index]
        if best_index != 0 and score_best > (0.9 * score_default):
            best_index = 0 

        best_raw_scores = all_scores[best_index]

        return best_index, best_raw_scores
    
    def run(self, original_np, inputs_np):
        """
        Runs the full OCS pipeline to determine the image with the best score.

        Args:
            original_np (np.ndarray): 
                Original RGB image, shape (H, W, 3), dtype=uint8, values in [0, 255].
            inputs_np (list[np.ndarray]): 
                List of candidate RGB input images. Images are expected to be in RGB format, shape (H, W, 3), dtype=uint8, values in [0, 255].

        Returns:
            best_image (np.ndarray): 
                The selected candidate image in RGB format, (H, W, 3), dtype=uint8, values in [0, 255].
        """
        # Compute reference boxes
        reference_boxes = self.detect_boxes(original_np)

        # Calculate object retention scores for each foggy image
        all_scores = []
        for image in inputs_np:
            foggy_boxes = self.detect_boxes(image)
            retention_score = self.evaluate_object_retention(foggy_boxes, reference_boxes)
            all_scores.append({'object_retention': retention_score})

        inputs_bgr_np = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in inputs_np]

        # Compute AuthESI and BRISQUE for each foggy image
        raw_scores = self.metric_evaluation_scores(inputs_bgr_np)

        # Combine the scores
        for i in range(len(inputs_np)):
            all_scores[i].update(raw_scores[i])

        # Determine the best image to return
        best_index, _ = self.find_best_result(all_scores)
        best_image = inputs_np[best_index]

        return best_image
