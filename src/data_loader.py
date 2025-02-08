from albumentation.pytorch import ToTensorV2
from torchvision.transforms import v2

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import h5py
import torch



def load_data(vol_range):
    images = []
    masks = []
    for vol_number in vol_range:
        for i in range(155):
            file_path = f'../data/raw/Brats2020/BraTS2020_training_data/content/data/volume_{vol_number}_slice_{i}.h5'
            with h5py.File(file_path, 'r') as f:
                image_data = f["image"][:]
                mask_data = f["mask"][:]
                

                images.append(image_data)
                masks.append(mask_data)


    images = np.array(images)
    masks = np.array(masks)
    return images, masks


def normalize_mri(images: np.ndarray, masks: np.ndarray, percentage_low=1, percentage_high=99) -> np.ndarray:
    """Normalize multi-modal MRI volumes"""
    # make sure all image values are represented by float32 for numerical stability
    images = images.astype(np.float32)

    # establish the percentiles per channel (channel is axis 3)
    p_low = np.percentile(images, percentage_low, axis=(0, 1, 2), keepdims=True)
    p_high = np.percentile(images, percentage_high, axis=(0, 1, 2), keepdims=True)

    # truncate all values of images to the low and high percentiles to get rid of outliers
    images = np.clip(images, p_low, p_high)
    images = (images - p_low) / (p_high - p_low)

    # the masks should simply be categorical
    masks = masks.astype(np.uint8)

    return images, masks


def convert_to_pytorch_tensors(images, masks):
    # Permute the array to (N, C, H, W) format where N=155, C=4 (modalities), H=240, W=240
    tensor_images = torch.from_numpy(images).permute(0, 3, 1, 2)

    # same for the mask
    tensor_masks = torch.from_numpy(masks).permute(0, 3, 1, 2)

    return tensor_images, tensor_masks


def transform(images, masks):
    tranforms = A.Compose([
        A.HorizontalFlip(p=0.5)
    ])


def preprocess_data(vol_range):
    images, masks = load_data(images, masks, vol_range)
    images, masks = normalize_mri(images, masks)
    tensor_images, tensor_masks = convert_to_pytorch_tensors(images, masks)
    return tensor_images, tensor_masks


if __name__ == "__main__":
    vol_range = range(1, 370)
    images, masks = preprocess_data(vol_range)