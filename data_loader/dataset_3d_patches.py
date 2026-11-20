from pathlib import Path
import re

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CT3DPatchDataset(Dataset):
    """
    Creates overlapping 3D patches from CT nodule slices.

    Each patch contains:
        image:        [1, patch_depth, H, W]
        mask:         [1, patch_depth, H, W]
        valid_slices: [patch_depth]

    Additional metadata:
        patient_id
        nodule_id
        start_idx

    start_idx allows predictions to be mapped back to their
    original slice positions.
    """

    def __init__(
        self,
        csv_file,
        image_root,
        mask_root,
        patch_depth=8,
        stride=4,
    ):
        self.data = pd.read_csv(csv_file)

        self.image_root = Path(image_root)
        self.mask_root = Path(mask_root)

        self.patch_depth = int(patch_depth)
        self.stride = int(stride)

        if self.patch_depth < 1 or self.stride < 1:
            raise ValueError(
                "patch_depth and stride must be positive."
            )

        required_columns = {"patient_id", "nodule_id"}

        missing_columns = required_columns - set(
            self.data.columns
        )

        if missing_columns:
            raise ValueError(
                f"Missing CSV columns: {missing_columns}"
            )

        self.samples = []

        for row in self.data.itertuples(index=False):

            patient_id = str(row.patient_id)
            nodule_id = str(row.nodule_id)

            image_dir = (
                self.image_root
                / patient_id
                / nodule_id
                / "images"
            )

            mask_dir = (
                self.mask_root
                / patient_id
                / nodule_id
            )

            if not image_dir.is_dir():
                continue

            image_paths = sorted(
                image_dir.glob("slice-*.png"),
                key=lambda p: int(
                    re.search(
                        r"slice-(\d+)",
                        p.stem,
                    ).group(1)
                ),
            )

            # Require at least two slices.
            if len(image_paths) < 2:
                continue

            # Require a matching mask for every image.
            if not all(
                (mask_dir / path.name).is_file()
                for path in image_paths
            ):
                continue

            depth = len(image_paths)

            if depth <= self.patch_depth:
                starts = [0]
            else:
                starts = list(
                    range(
                        0,
                        depth - self.patch_depth + 1,
                        self.stride,
                    )
                )

                final_start = depth - self.patch_depth

                if starts[-1] != final_start:
                    starts.append(final_start)

            for start in starts:

                selected_paths = image_paths[
                    start:start + self.patch_depth
                ]

                self.samples.append(
                    {
                        "patient_id": patient_id,
                        "nodule_id": nodule_id,
                        "image_paths": selected_paths,
                        "mask_dir": mask_dir,
                        "start_idx": start,
                        "real_depth": len(selected_paths),
                    }
                )

        if not self.samples:
            raise RuntimeError(
                "No valid 3D patches were created. "
                "Check the CSV and dataset directories."
            )

        print(
            f"Created {len(self.samples)} patches "
            f"from {len(self.data)} nodule records."
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

            if image is None:
                raise RuntimeError(
                    f"Could not read image: {image_path}"
                )

            if mask is None:
                raise RuntimeError(
                    f"Could not read mask: {mask_path}"
                )

            if image.shape != mask.shape:
                mask = cv2.resize(
                    mask,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            # Keep preprocessing consistent with training.
            image = image.astype(np.float32) / 255.0
            mask = (mask > 0).astype(np.float32)

            images.append(image)
            masks.append(mask)

        volume = np.stack(images, axis=0)
        mask_volume = np.stack(masks, axis=0)

        real_depth = volume.shape[0]
        pad_depth = self.patch_depth - real_depth

        valid_slices = np.zeros(
            self.patch_depth,
            dtype=np.float32,
        )

        valid_slices[:real_depth] = 1.0

        if pad_depth > 0:

            volume = np.pad(
                volume,
                (
                    (0, pad_depth),
                    (0, 0),
                    (0, 0),
                ),
                mode="constant",
                constant_values=0,
            )

            mask_volume = np.pad(
                mask_volume,
                (
                    (0, pad_depth),
                    (0, 0),
                    (0, 0),
                ),
                mode="constant",
                constant_values=0,
            )

        volume = torch.from_numpy(
            np.ascontiguousarray(volume)
        ).unsqueeze(0)

        mask_volume = torch.from_numpy(
            np.ascontiguousarray(mask_volume)
        ).unsqueeze(0)

        valid_slices = torch.from_numpy(valid_slices)

        return {
            "image": volume,
            "mask": mask_volume,
            "valid_slices": valid_slices,
            "patient_id": sample["patient_id"],
            "nodule_id": sample["nodule_id"],
            "start_idx": sample["start_idx"],
            "real_depth": sample["real_depth"],
        }