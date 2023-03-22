import torch
import torchvision
import numpy as np
from functools import partial
import torchvision.transforms.functional as F
import torchvision.transforms as transforms


def add_mask(image, mask_percentage, inpainting=False):
    """Add a mask to an image.
    Args:
        image (PIL.Image): Image to add the mask to.
        mask_percentage (float): Percentage of the image to mask.
        inpainting (bool): If True, the mask will be a square in the center of the image.
    Returns:
            torch.Tensor: Image with the mask."""
    image = F.to_tensor(image)
    num_channels, height, width = image.shape
    mask = torch.ones((height, width), dtype=torch.bool)
    mask_width = int(width * mask_percentage)
    random_delta = np.random.randint(-4, 5)
    if inpainting:
        mask_size = int(np.sqrt(np.prod(image.shape[1:]) * mask_percentage))
        start_h = (image.shape[1] - mask_size) // 2
        start_w = (image.shape[2] - mask_size) // 2
        mask[start_h : start_h + mask_size, start_w : start_w + mask_size] = 0
    else:
        mask_width = mask_width + random_delta
        mask[:, width - mask_width :] = 0
    return torch.cat([image, mask.reshape(1, height, width)], dim=0)


class MinMaxScaling:
    def __call__(self, image):
        image = F.to_tensor(image)
        min_val, max_val = image.min(), image.max()
#         min_val, max_val = 0, 255
        image = (image - min_val) / (max_val - min_val)
        return F.to_pil_image(2 * image - 1)

class AddMask:
    def __init__(self, mask_percentage, inpainting=False):
        self.mask_percentage = mask_percentage
        self.inpainting = inpainting

    def __call__(self, image):
        return add_mask(image, self.mask_percentage, self.inpainting)


# Define the transformation pipeline
transform_pipeline = transforms.Compose([
    MinMaxScaling(),
    AddMask(mask_percentage=0.25, inpainting=False),
])

# Usage example
places365_dataset = torchvision.datasets.Places365(
    "./data/pt_dataset/",
    small=True,
    download=False,
    transform=transform_pipeline,
)
