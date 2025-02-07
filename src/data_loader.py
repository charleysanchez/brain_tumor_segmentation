import matplotlib.pyplot as plt
import numpy as np
import h5py



def load_data():
    images = []
    masks = []
    for i in range(155):
        file_path = f'../data/raw/Brats2020/BraTS2020_training_data/content/data/volume_1_slice_{i}.h5'
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

images, masks = load_data()
images, masks = normalize_mri(images, masks)