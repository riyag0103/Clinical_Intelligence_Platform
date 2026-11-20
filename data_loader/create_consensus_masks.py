from pathlib import Path
from PIL import Image
import numpy as np
import re

# Dataset location
DATASET_ROOT = Path(
    r"D:\Clinical_Project\datasets\raw\ct\LIDC-IDRI-slices"
)

# Save generated masks separately; never overwrite original annotations.
OUTPUT_ROOT = Path(
    r"D:\Clinical_Project\datasets\processed\lidc_consensus_masks"
)


def natural_key(path):
    """Sort slice-2 before slice-10."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def load_binary_mask(mask_path):
    """Load a mask and convert it to a binary array."""
    with Image.open(mask_path) as image:
        array = np.asarray(image.convert("L"))

    return array > 0


def main():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_ROOT}"
        )

    nodule_dirs = sorted(
        path
        for path in DATASET_ROOT.rglob("nodule-*")
        if path.is_dir() and (path / "images").is_dir()
    )

    processed_slices = 0
    skipped_slices = 0
    positive_slices = 0

    for nodule_dir in nodule_dirs:
        image_dir = nodule_dir / "images"

        mask_dirs = sorted(
            path
            for path in nodule_dir.glob("mask-*")
            if path.is_dir()
        )

        if not mask_dirs:
            print(f"Skipping: no masks in {nodule_dir}")
            continue

        images = sorted(
            image_dir.glob("*.png"),
            key=natural_key
        )

        # Preserve patient/nodule folder organization.
        relative_path = nodule_dir.relative_to(DATASET_ROOT)
        output_dir = OUTPUT_ROOT / relative_path
        output_dir.mkdir(parents=True, exist_ok=True)

        for image_path in images:
            available_masks = []

            for mask_dir in mask_dirs:
                mask_path = mask_dir / image_path.name

                if not mask_path.exists():
                    continue

                mask = load_binary_mask(mask_path)

                with Image.open(image_path) as image:
                    image_size = image.size

                if (
                    mask.shape[1] != image_size[0]
                    or mask.shape[0] != image_size[1]
                ):
                    raise ValueError(
                        f"Image-mask dimension mismatch: {mask_path}"
                    )

                available_masks.append(mask)

            if not available_masks:
                skipped_slices += 1
                continue

            shapes = {mask.shape for mask in available_masks}

            if len(shapes) != 1:
                raise ValueError(
                    f"Inconsistent mask shapes for {image_path}"
                )

            # Majority vote among the available annotations.
            votes = np.stack(
                available_masks,
                axis=0
            ).sum(axis=0)

            consensus = (
                votes >= (len(available_masks) / 2)
            ).astype(np.uint8) * 255

            output_path = output_dir / image_path.name

            Image.fromarray(consensus).save(output_path)

            processed_slices += 1

            if np.any(consensus):
                positive_slices += 1

    print("\n========== CONSENSUS MASK GENERATION ==========")
    print(f"Nodule folders found: {len(nodule_dirs)}")
    print(f"Slices processed: {processed_slices}")
    print(f"Slices skipped: {skipped_slices}")
    print(f"Slices with positive consensus masks: {positive_slices}")
    print(f"Output directory: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()