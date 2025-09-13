import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

from .SCI import Finetunemodel
from .DHAN import Deshadower


# Need to download SRD+ models from:
#   https://drive.google.com/uc?id=1rEIWWLwEpbZGPyFUc9jSIQr78ZeQy5eZ
# And VGG-19 model from:
#   https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat
# Place both in ./DHAN/models
class ImageEnhancer:
    def __init__(self, device="cuda"):
        """
        Initialise ImageEnhancer with the specified device and load the brightening and deshadowing models.

        Args:
            device (str):
                Device to run the model on: "cuda", "cpu" or "mps" (default: "cuda").
        """
        self.width_resize = 640
        self.height_resize = 480

        if device == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Model checkpoint paths
        checkpoint_bright = "./src/fog_pipeline/image_enhancement/SCI/easy.pt"
        srdplus_pretrained = "./src/fog_pipeline/image_enhancement/DHAN/models"
        vgg_19_path = "./src/fog_pipeline/image_enhancement/DHAN/models/imagenet-vgg-verydeep-19.mat"

        # Load brightening model
        self.model_bright = Finetunemodel(checkpoint_bright)
        self.model_bright = self.model_bright.to(self.device)
        self.model_bright.eval()

        # Load deshadowing model
        use_gpu = 0 if self.device.type == "cuda" else -1
        self.model_deshadow = Deshadower(
            model_path=srdplus_pretrained, 
            vgg_19_path=vgg_19_path, 
            use_gpu=use_gpu, 
            hyper=1
        )
    
    def prepare_image_deshadow(self, img):
        """
        Prepare an RGB image for deshadowing by converting to BGR, resizing to a fixed height and normalising the pixel values.

        Adapted from DHAN:
            https://github.com/vinthony/ghost-free-shadow-removal

        Args:
            img (np.ndarray): 
                Input RGB image as a NumPy array, shape (H, W, 3), dtype=uint8, values in [0, 255].
        
        Returns:
            resized_img (np.ndarray):
                Resized and normalised BGR image, shape (self.height_resize, width_resize, 3), dtype=float32, values in [0, 1].
        """
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Calculate resize width to keep aspect ratio with fixed height
        self.width_resize = int(img.shape[1] * self.height_resize / float(img.shape[0]))

        # Resize and normalise pixel values to [0, 1]
        resized_img = cv2.resize(np.float32(img), (self.width_resize, self.height_resize), cv2.INTER_CUBIC) / 255.0

        return resized_img
    
    def image_partition(self, input_img_path):
        """
        Partition an RGB image into overlapping tiles for patch-wise processing.

        Args:
            input_img_path (str): 
                Path to an input RGB image file. Images are expected to be in RGB format, shape (H, W, 3), dtype=uint8, values in [0, 255].

        Returns:
            all_partitions (List[np.ndarray]):
                List of RGB image tiles, each shape (H/2, W/2, 3), dtype=uint8, values in [0, 255].
        """
        image = Image.open(input_img_path).convert("RGB")
        image_width, image_height = image.size

        # Tile size is half image size with 50% overlap
        # Stride is half the tile size
        tile_width = image_width // 2
        tile_height = image_height // 2
        stride_x = tile_width // 2
        stride_y = tile_height // 2

        all_partitions = []
        for y in range(0, image_height - tile_height + 1, stride_y):
            for x in range(0, image_width - tile_width + 1, stride_x):
                crop_partition = (x, y, x + tile_width, y + tile_height)
                
                partition_pil = image.crop(crop_partition)
                partition = np.array(partition_pil)

                all_partitions.append(partition)

        return all_partitions

    def horizontal_blend(self, img_left, img_right, overlap_start, overlap_width):
        """
        Blend two images horizontally with a linear alpha mask on the overlap region.

        Args:
            img_left (np.ndarray):
                Left RGB image, shape (H, W, 3), dtype=float32, values in [0, 1].
            img_right (np.ndarray):
                Right RGB image, shape (H, W, 3), dtype=float32, values in [0, 1].
            overlap_start (int):
                Horizontal start index of overlap region.
            overlap_width (int):
                Width of the overlap region.

        Returns:
            result (np.ndarray):
                Horizontally blended RGB image, dtype=float32, values in [0, 1].
        """
        h = img_left.shape[0]
        w_total = img_left.shape[1] + img_right.shape[1] - overlap_width
        c = img_left.shape[2]
        result = np.zeros((h, w_total, c), dtype=np.float32)

        # Copy the pixels of the left image when there is no overlap and the current pixel is within the left image
        result[:, :overlap_start, :] = img_left[:, :overlap_start, :]

        # Linearly blend the pixels of both images in the overlap area
        alpha = np.linspace(0, 1, overlap_width, endpoint=False).reshape(1, -1, 1)
        result[:, overlap_start : overlap_start + overlap_width, :] = (
            img_left[:, overlap_start : overlap_start + overlap_width, :] * (1 - alpha)
            + img_right[:, :overlap_width, :] * alpha
        )

        # Copy the pixels of the right image when there is no overlap and the current pixel is within the right image
        result[:, overlap_start + overlap_width :, :] = img_right[:, overlap_width:, :]

        return result

    def vertical_blend(self, img_top, img_bottom, overlap_start, overlap_height):
        """
        Blend two images vertically with a linear alpha mask on the overlap region.

        Args:
            img_top (np.ndarray):
                Top RGB image, shape (H, W, 3), dtype=float32, values in [0, 1].
            img_bottom (np.ndarray):
                Bottom RGB image, shape (H, W, 3), dtype=float32, values in [0, 1].
            overlap_start (int):
                Vertical start index of overlap region.
            overlap_height (int):
                Height of the overlap region.

        Returns:
            result (np.ndarray):
                Vertically blended RGB image, dtype=float32, values in [0, 1].
        """
        h_total = img_top.shape[0] + img_bottom.shape[0] - overlap_height
        w = img_top.shape[1]
        c = img_top.shape[2]
        result = np.zeros((h_total, w, c), dtype=np.float32)

        # Copy the pixels of the top image when there is no overlap and the current pixel is within the top image
        result[:overlap_start, :, :] = img_top[:overlap_start, :, :]

        # Linearly blend the pixels of both images in the overlap area
        alpha = np.linspace(0, 1, overlap_height, endpoint=False).reshape(-1, 1, 1)
        result[overlap_start : overlap_start + overlap_height, :, :] = (
            img_top[overlap_start : overlap_start + overlap_height, :, :] * (1 - alpha)
            + img_bottom[:overlap_height, :, :] * alpha
        )

        # Copy the pixels of the bottom image when there is no overlap and the current pixel is within the bottom image
        result[overlap_start + overlap_height :, :, :] = img_bottom[overlap_height:, :]

        return result

    def stitch_partitions(self, partitions, tile_width, tile_height, stride_x, stride_y):
        """
        Stitch a grid of overlapping image partitions back into a full image.

        Args:
            partitions (List[np.ndarray]): 
                List of image partitions, shape (tile_H, tile_W, 3), dtype=float32, values in [0, 1].
            tile_width (int): 
                Width of each tile.
            tile_height (int): 
                Height of each tile.
            stride_x (int): 
                Horizontal stride (overlap step).
            stride_y (int): 
                Vertical stride (overlap step).

        Returns:
            stitched (np.ndarray): 
                Fully stitched RGB image, dtype=float32, values in [0, 1].
        """
        num_tiles = len(partitions)
        grid_size = int(np.sqrt(num_tiles))
        
        rows = []

        # Stitch the tiles horizontally per row
        for row_i in range(grid_size):
            row_tiles = partitions[row_i * grid_size : (row_i + 1) * grid_size]
            row = row_tiles[0]
            for i in range(1, len(row_tiles)):
                overlap_start = stride_x * i
                overlap_width = tile_width - stride_x
                row = self.horizontal_blend(row, row_tiles[i], overlap_start, overlap_width)
            rows.append(row)

        # Stitch the horizontal rows vertically
        stitched = rows[0]
        for i in range(1, len(rows)):
            overlap_start = stride_y * i
            overlap_height = tile_height - stride_y
            stitched = self.vertical_blend(stitched, rows[i], overlap_start, overlap_height)

        return stitched

    def deshadower(self, partitions):
        """
        Apply the deshadowing model to each partition and stitch the results into a full image.

        Adapted from DHAN:
            https://github.com/vinthony/ghost-free-shadow-removal

        Args:
            partitions (List[np.ndarray]):
                List of RGB image partitions, shape (H/2, W/2, 3), dtype=uint8, values in [0, 255].

        Returns:
            stitched_clipped (np.ndarray):
                Fully stitched deshadowed RGB image, shape (H, W, 3), dtype=uint8, values in [0, 255].
        """
        outputs = []
        
        for image in partitions:
            # Prepare image for deshadowing
            prepared_img = self.prepare_image_deshadow(image)

            # Run deshadowing model
            output_deshadow_np = self.model_deshadow.run(prepared_img)

            # Convert BGR output to RGB for consistency
            output_deshadow_np = cv2.cvtColor(output_deshadow_np, cv2.COLOR_BGR2RGB)

            outputs.append(output_deshadow_np)

        # Stitched deshadowed partitions into a full image
        tile_h, tile_w = outputs[0].shape[:2]
        stride_x = tile_w // 2
        stride_y = tile_h // 2
        stitched = self.stitch_partitions(outputs, tile_w, tile_h, stride_x, stride_y)

        stitched_clipped = np.clip(stitched, 0, 255).astype(np.uint8)

        return stitched_clipped

    @torch.no_grad()
    def brightener(self, partitions):
        """
        Apply the brightening model to each partition and stitch the results into a full image.

        Adapted from SCI:
            https://github.com/vis-opt-group/SCI
            Copyright (c) 2022 Tengyu Ma. MIT License
            See LICENSES/SCI for the full license text

        Args:
            partitions (List[np.ndarray]): 
                List of RGB image partitions, shape (H/2, W/2, 3), dtype=uint8, values in [0, 255].

        Returns:
            stitched_clipped (np.ndarray):
                Fully stitched brightened RGB image, shape (H, W, 3), dtype=uint8, values in [0, 255].
        """
        outputs = []
        
        for image in partitions:
            # Transform NumPy image to tensor
            img_pil = Image.fromarray(image)
            img_tensor = transforms.ToTensor()(img_pil).unsqueeze(0).to(self.device)
                
            # Run brightening model
            _, output_bright = self.model_bright(img_tensor)

            # Convert output from tensor to uint8 image (H, W, C)
            output_bright_np = np.clip(output_bright[0].cpu().numpy(), 0, 1)
            output_bright_np = (output_bright_np * 255).astype(np.uint8)
            output_bright_np = np.transpose(output_bright_np, (1, 2, 0))

            outputs.append(output_bright_np)

        # Stitch brightened partitions into a full image
        tile_h, tile_w = outputs[0].shape[:2]
        stride_x = tile_w // 2
        stride_y = tile_h // 2

        stitched = self.stitch_partitions(outputs, tile_w, tile_h, stride_x, stride_y)
        stitched_clipped = np.clip(stitched, 0, 255).astype(np.uint8)

        return stitched_clipped

    def run(self, input_img_path):
        """
        Run the full image enhancement pipeline by applying the different enhancements.

        Args:
            input_img_path (str): 
                Path to the input RGB image. Image expected as dtype=uint8, shape (H, W, 3), values in [0, 255].

        Returns:
            output_deshadowed (np.ndarray):
                Deshadowed RGB image, shape (H, W, 3), dtype=uint8, values in [0, 255].
            output_brightened (np.ndarray):
                Brightened RGB image, shape (H, W, 3), dtype=uint8, values in [0, 255].
        """
        # Partition the input image into overlapping tiles
        partitions = self.image_partition(input_img_path)

        # Deshadow partitions and stitch
        output_deshadowed = self.deshadower(partitions)

        # Brighten partitions and stitch
        output_brightened = self.brightener(partitions)

        # Load original image to get its shape
        original_img = np.array(Image.open(input_img_path).convert("RGB"))
        original_h, original_w = original_img.shape[:2]

        # Resize outputs to match original shape
        output_deshadowed = cv2.resize(output_deshadowed, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
        output_brightened = cv2.resize(output_brightened, (original_w, original_h), interpolation=cv2.INTER_CUBIC)

        return output_deshadowed, output_brightened
    