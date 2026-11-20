from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(r"D:\Clinical_Project")
sys.path.insert(0, str(PROJECT_ROOT))

from data_loader.dataset_3d import CT3DSegmentationDataset


dataset = CT3DSegmentationDataset(
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
    target_size=(128, 128),
    min_slices=2,
)

print("\nTotal nodule stacks:", len(dataset))

# Test a few samples with different depths.
for idx in [0, len(dataset) // 2, len(dataset) - 1]:
    sample = dataset[idx]

    image = sample["image"]
    mask = sample["mask"]

    print("\nPatient:", sample["patient_id"])
    print("Nodule:", sample["nodule_id"])
    print("Number of slices:", sample["num_slices"])
    print("Image shape:", tuple(image.shape))
    print("Mask shape:", tuple(mask.shape))
    print("Image range:", image.min().item(), image.max().item())
    print("Mask values:", torch.unique(mask).tolist())
    print("Positive mask voxels:", int(mask.sum().item()))

print("\n3D dataset loader test completed.")