from pathlib import Path
from PIL import Image
import numpy as np

IMAGE_ROOT = Path(
    r"D:\Clinical_Project\datasets\raw\ct\LIDC-IDRI-slices"
)

MASK_ROOT = Path(
    r"D:\Clinical_Project\datasets\processed\lidc_consensus_masks"
)


def main():
    total_masks = 0
    missing_images = 0
    dimension_mismatches = 0
    invalid_values = 0
    empty_masks = 0
    positive_masks = 0

    for mask_path in MASK_ROOT.rglob("*.png"):
        total_masks += 1

        relative_path = mask_path.relative_to(MASK_ROOT)

        image_path = (
            IMAGE_ROOT
            / relative_path.parent
            / "images"
            / relative_path.name
        )

        if not image_path.exists():
            missing_images += 1
            continue

        with Image.open(image_path) as image:
            image_size = image.size

        with Image.open(mask_path) as mask_image:
            mask = np.asarray(mask_image.convert("L"))

        if mask.shape[::-1] != image_size:
            dimension_mismatches += 1

        if not np.isin(mask, [0, 255]).all():
            invalid_values += 1

        if np.any(mask):
            positive_masks += 1
        else:
            empty_masks += 1

    print("\n========== CONSENSUS MASK VALIDATION ==========")
    print(f"Total consensus masks: {total_masks}")
    print(f"Missing corresponding images: {missing_images}")
    print(f"Dimension mismatches: {dimension_mismatches}")
    print(f"Masks with unexpected pixel values: {invalid_values}")
    print(f"Positive masks: {positive_masks}")
    print(f"Empty masks: {empty_masks}")

    if (
        missing_images == 0
        and dimension_mismatches == 0
        and invalid_values == 0
    ):
        print("\nBasic validation passed.")
    else:
        print("\nValidation found issues that need investigation.")


if __name__ == "__main__":
    main()