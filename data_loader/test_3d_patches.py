from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(r"D:\Clinical_Project")
sys.path.insert(0, str(PROJECT_ROOT))

from data_loader.dataset_3d_patches import CT3DPatchDataset


dataset = CT3DPatchDataset(
    csv_file=(
        PROJECT_ROOT
        / "datasets"
        / "metadata"
        / "3d_splits"
        / "train_3d.csv"
    ),
    image_root=(
        PROJECT_ROOT
        / "datasets"
        / "raw"
        / "ct"
        / "LIDC-IDRI-slices"
    ),
    mask_root=(
        PROJECT_ROOT
        / "datasets"
        / "processed"
        / "lidc_consensus_masks"
    ),
    patch_depth=8,
    stride=4,
)

positive_patches = 0
empty_patches = 0

print("\nTotal patches:", len(dataset))

for idx in range(len(dataset)):

    sample = dataset[idx]

    image = sample["image"]
    mask = sample["mask"]
    valid_slices = sample["valid_slices"]

    if mask.sum().item() > 0:
        positive_patches += 1
    else:
        empty_patches += 1

    assert image.shape == (1, 8, 128, 128)
    assert mask.shape == (1, 8, 128, 128)
    assert valid_slices.shape == (8,)

    assert torch.all(
        (valid_slices == 0) | (valid_slices == 1)
    )

    if idx < 3:
        print("\nPatient:", sample["patient_id"])
        print("Nodule:", sample["nodule_id"])
        print("Image shape:", tuple(image.shape))
        print("Mask shape:", tuple(mask.shape))
        print("Valid slices:", valid_slices.tolist())
        print(
            "Positive voxels:",
            int(mask.sum().item()),
        )

print("\nPositive patches:", positive_patches)
print("Empty patches:", empty_patches)
print(
    "Positive patch percentage:",
    round(100 * positive_patches / len(dataset), 2),
)

print("\n3D patch validation passed.")