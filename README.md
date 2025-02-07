# Brain Tumor Segmentation using the BraTS 2020 Dataset

## Project Overview
This project focuses on brain tumor segmentation using deep learning techniques, specifically using the **BraTS 2020** dataset. The dataset consists of MRI scans of brain tumors, with each scan containing multiple image modalities, including T1, T1Gd, T2, and FLAIR. The goal is to accurately segment different tumor regions, such as the **tumor core** and **enhancing tumor**, to aid in the diagnosis and treatment planning for brain cancer patients.

## Dataset
The dataset used in this project is from the **BraTS 2020** challenge. It includes images from multiple brain scans, with annotations (masks) for various tumor regions. Each volume consists of 3D MRI scans, with slices corresponding to different depths of the brain.

### Data Modalities:
- **T1**: Structural imaging of the brain.
- **T1Gd**: Post-contrast T1-weighted image, highlighting the enhancing regions of the tumor.
- **T2**: Provides contrast for edema and other tissue anomalies.
- **FLAIR**: Suppresses cerebrospinal fluid to improve the detection of lesions and tumors.

### Mask Labels:
- **Background**: No tumor.
- **Tumor Core**: The central part of the tumor, often necrotic or exhibiting edema.
- **Enhancing Tumor**: Tumor regions that enhance with contrast, indicating high vascularization and aggressive growth.

## Installation

To set up the project environment, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/brain-tumor-segmentation.git
   cd brain-tumor-segmentation
