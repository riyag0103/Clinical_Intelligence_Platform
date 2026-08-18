import cv2
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset

# ----------------------------------
# CT SEGMENTATION DATASET
# ----------------------------------
class CTSegmentationDataset(Dataset):

    def __init__(self, csv_file, transforms=None):

        # Load metadata CSV
        self.data = pd.read_csv(csv_file)

        # Augmentations
        self.transforms = transforms

    # ----------------------------------
    # TOTAL SAMPLES
    # ----------------------------------
    def __len__(self):

        return len(self.data)

    # ----------------------------------
    # LOAD SINGLE SAMPLE
    # ----------------------------------
    def __getitem__(self, idx):

        # ----------------------------------
        # IMAGE PATH
        # ----------------------------------
        image_path = self.data.iloc[idx]["image_path"]

        # ----------------------------------
        # MASK PATH
        # ----------------------------------
        mask_path = self.data.iloc[idx]["mask_path"]

        # ----------------------------------
        # LOAD IMAGE
        # ----------------------------------
        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        # ----------------------------------
        # LOAD MASK
        # ----------------------------------
        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        # ----------------------------------
        # CHECK IF FILES LOADED
        # ----------------------------------
        if image is None:
            raise ValueError(
                f"Image not found: {image_path}"
            )

        if mask is None:
            raise ValueError(
                f"Mask not found: {mask_path}"
            )

        # ----------------------------------
        # FORCE FIXED SIZE
        # ----------------------------------
        image = cv2.resize(
            image,
            (256, 256),
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            (256, 256),
            interpolation=cv2.INTER_NEAREST
        )

        # ----------------------------------
        # NORMALIZE
        # ----------------------------------
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        # ----------------------------------
        # AUGMENTATIONS
        # ----------------------------------
        if self.transforms:

            augmented = self.transforms(
                image=image,
                mask=mask
            )

            image = augmented["image"]
            mask = augmented["mask"]

        # ----------------------------------
        # CHANNEL DIMENSION
        # ----------------------------------
        image = np.expand_dims(
            image,
            axis=0
        )

        mask = np.expand_dims(
            mask,
            axis=0
        )

        # ----------------------------------
        # TO TENSOR
        # ----------------------------------
        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        mask = torch.tensor(
            mask,
            dtype=torch.float32
        )

        return image, mask