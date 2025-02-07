import matplotlib.pyplot as plt
import numpy as np
import h5py

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
print("shape of image:", images.shape)
print("shape of mask:", masks.shape)

images = images.astype(np.float32)
