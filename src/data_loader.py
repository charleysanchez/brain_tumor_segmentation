from albumentations.pytorch import ToTensorV2
from torchvision.transforms import v2

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import h5py
import torch



def load_data(vol_range):
    images = []
    masks = []
    for vol_number in range(1, vol_range):
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


def convert_to_pytorch_tensors(images, masks, verbose=True):
    if verbose:
        print("working on convert to pytorch tensor")

    # Permute the array to (N, C, H, W) format where N=155, C=4 (modalities), H=240, W=240
    tensor_images = torch.from_numpy(images).permute(0, 3, 1, 2)

    # same for the mask
    tensor_masks = torch.from_numpy(masks).permute(0, 3, 1, 2)

    return tensor_images, tensor_masks


def transform(images, masks, verbose=True):
    if verbose:
        print("working on transform")

    transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])

    transformed_images = []
    transformed_masks = []

    for image, mask in zip(images, masks):
        transformed = transforms(image=image.astype(np.float32), mask=mask.astype(np.uint8))
        transformed_images.append(transformed["image"])
        transformed_masks.append(transformed["mask"])
    return transformed_images, transformed_masks 


def preprocess_data(vol_range=370):
    images, masks = load_data(vol_range)
    # tensor_images, tensor_masks = convert_to_pytorch_tensors(images, masks)
    image_transformed, mask_transformed = transform(images, masks)
    return image_transformed, mask_transformed


if __name__ == "__main__":
    vol_range = range(1, 370)
    images, masks = preprocess_data(vol_range)