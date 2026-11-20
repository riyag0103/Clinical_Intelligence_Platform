from pathlib import Path

import cv2
import numpy as np
import torch

from torch.utils.data import Dataset


class CT3DSegmentationDataset(Dataset):
    """
    Loads one complete nodule stack as a 3D segmentation sample.

    Image output: [1, D, H, W]
    Mask output:  [1, D, H, W]

    D varies between nodules.
    """

    def __init__(
        self,
        image_root,
        mask_root,
        target_size=(128, 128),
        min_slices=2,
    ):
        self.image_root = Path(image_root)
        self.mask_root = Path(mask_root)
        self.target_size = target_size
        self.min_slices = min_slices

        if not self.image_root.is_dir():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_root}"
            )

        if not self.mask_root.is_dir():
            raise FileNotFoundError(
                f"Mask directory not found: {self.mask_root}"
            )

        self.samples = []

        for patient_dir in sorted(
            self.image_root.glob("LIDC-IDRI-*")
        ):
            for nodule_dir in sorted(
                patient_dir.glob("nodule-*")
            ):
                image_dir = nodule_dir / "images"

                if not image_dir.is_dir():
                    continue

                mask_dir = (
                    self.mask_root
                    / patient_dir.name
                    / nodule_dir.name
                )

                image_paths = sorted(
                    image_dir.glob("slice-*.png"),
                    key=lambda p: int(
                        p.stem.split("-")[-1]
                    ),
                )

                if len(image_paths) < self.min_slices:
                    continue

                # Keep only stacks with a corresponding
                # consensus mask for every slice.
                if not all(
                    (mask_dir / p.name).is_file()
                    for p in image_paths
                ):
                    continue

                self.samples.append({
                    "patient_id": patient_dir.name,
                    "nodule_id": nodule_dir.name,
                    "image_paths": image_paths,
                    "mask_dir": mask_dir,
                })

        if not self.samples:
            raise RuntimeError(
                "No valid nodule stacks were found."
            )

        print(
            f"Loaded {len(self.samples)} nodule stacks "
            f"with at least {self.min_slices} slices."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        images = []
        masks = []

        for image_path in sample["image_paths"]:
            mask_path = sample["mask_dir"] / image_path.name

            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_GRAYSCALE,
            )

            mask = cv2.imread(
                str(mask_path),
                cv2.IMREAD_GRAYSCALE,
            )

            if image is None or mask is None:
                raise RuntimeError(
                    f"Could not read image or mask: {image_path}"
                )

            if image.shape != mask.shape:
                raise RuntimeError(
                    f"Image-mask dimension mismatch: {image_path}"
                )

            # Keep image and mask spatially aligned.
            image = cv2.resize(
                image,
                self.target_size,
                interpolation=cv2.INTER_LINEAR,
            )

            mask = cv2.resize(
                mask,
                self.target_size,
                interpolation=cv2.INTER_NEAREST,
            )

            image = image.astype(np.float32) / 255.0
            mask = (mask > 0).astype(np.float32)

            images.append(image)
            masks.append(mask)

        # Stack slices along the depth dimension.
        volume = np.stack(images, axis=0)
        mask_volume = np.stack(masks, axis=0)

        # [D, H, W] -> [1, D, H, W]
        volume = torch.from_numpy(volume).unsqueeze(0)
        mask_volume = torch.from_numpy(
            mask_volume
        ).unsqueeze(0)

        return {
            "image": volume,
            "mask": mask_volume,
            "patient_id": sample["patient_id"],
            "nodule_id": sample["nodule_id"],
            "num_slices": len(images),
        }