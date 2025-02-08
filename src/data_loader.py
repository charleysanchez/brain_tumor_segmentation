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


def convert_to_pytorch_tensors(images, masks):
    # Permute the array to (N, C, H, W) format where N=155, C=4 (modalities), H=240, W=240
    tensor_images = torch.from_numpy(images).permute(0, 3, 1, 2)

    # same for the mask
    tensor_masks = torch.from_numpy(masks).permute(0, 3, 1, 2)

    return tensor_images, tensor_masks


def transform(images, masks):
    transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
        A.GaussNoise(var_limit=(0.01, 0.05), p=0.2),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])

    transformed = transforms(image=images, mask=masks)
    return transformed["image"], transformed["mask"]


def preprocess_data(vol_range=370):
    images, masks = load_data(vol_range)
    tensor_images, tensor_masks = convert_to_pytorch_tensors(images, masks)
    image_transformed, mask_transformed = transform(tensor_images, tensor_masks)
    return image_transformed, mask_transformed


if __name__ == "__main__":
    vol_range = range(1, 370)
    images, masks = preprocess_data(vol_range)