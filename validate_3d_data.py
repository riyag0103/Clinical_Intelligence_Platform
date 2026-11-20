from pathlib import Path
import cv2

PROJECT_ROOT = Path(r"D:\Clinical_Project")

IMAGE_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "ct"
    / "LIDC-IDRI-slices"
)

MASK_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "lidc_consensus_masks"
)


def main():
    print("=" * 60)
    print("VALIDATING ORIGINAL CT IMAGES AND CONSENSUS MASKS")
    print("=" * 60)

    total = 0
    valid = 0
    missing_images = 0
    missing_masks = 0
    dimension_errors = 0
    unreadable = 0

    for patient_dir in sorted(IMAGE_ROOT.glob("LIDC-IDRI-*")):
        for nodule_dir in sorted(patient_dir.glob("nodule-*")):

            image_dir = nodule_dir / "images"

            if not image_dir.is_dir():
                continue

            consensus_dir = (
                MASK_ROOT / patient_dir.name / nodule_dir.name
            )

            for image_path in sorted(image_dir.glob("slice-*.png")):
                total += 1

                mask_path = consensus_dir / image_path.name

                if not image_path.is_file():
                    missing_images += 1
                    continue

                if not mask_path.is_file():
                    missing_masks += 1
                    continue

                image = cv2.imread(
                    str(image_path), cv2.IMREAD_GRAYSCALE
                )
                mask = cv2.imread(
                    str(mask_path), cv2.IMREAD_GRAYSCALE
                )

                if image is None or mask is None:
                    unreadable += 1
                    continue

                if image.shape != mask.shape:
                    dimension_errors += 1
                    continue

                valid += 1

    print(f"\nTotal original CT slices: {total}")
    print(f"Valid image-mask pairs:   {valid}")
    print(f"Missing images:           {missing_images}")
    print(f"Missing consensus masks:  {missing_masks}")
    print(f"Dimension mismatches:     {dimension_errors}")
    print(f"Unreadable files:         {unreadable}")

    if (
        total > 0
        and valid == total
        and missing_images == 0
        and missing_masks == 0
        and dimension_errors == 0
        and unreadable == 0
    ):
        print("\nSUCCESS: All image-mask pairs passed validation.")
        print("Next step: Prepare the 3D dataset.")
    else:
        print("\nWARNING: Review the validation results before training.")


if __name__ == "__main__":
    main()