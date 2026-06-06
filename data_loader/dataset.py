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
        # NORMALIZE IMAGE
        # ----------------------------------
        image = image.astype(np.float32) / 255.0

        # ----------------------------------
        # NORMALIZE MASK
        # ----------------------------------
        mask = mask.astype(np.float32) / 255.0

        # ----------------------------------
        # APPLY AUGMENTATIONS
        # ----------------------------------
        if self.transforms:

            augmented = self.transforms(
                image=image,
                mask=mask
            )

            image = augmented["image"]
            mask = augmented["mask"]

        # ----------------------------------
        # ADD CHANNEL DIMENSION
        # ----------------------------------
        image = np.expand_dims(image, axis=0)

        mask = np.expand_dims(mask, axis=0)

        # ----------------------------------
        # CONVERT TO TENSORS
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