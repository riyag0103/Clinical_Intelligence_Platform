from pathlib import Path
import json
import random

import pandas as pd
from sklearn.model_selection import train_test_split


# ==================================================
# CONFIGURATION
# ==================================================

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "metadata"
    / "3d_splits"
)

SEED = 42


# ==================================================
# BUILD NODULE METADATA
# ==================================================

def build_metadata():

    records = []

    for patient_dir in sorted(
        IMAGE_ROOT.glob("LIDC-IDRI-*")
    ):
        for nodule_dir in sorted(
            patient_dir.glob("nodule-*")
        ):
            image_dir = nodule_dir / "images"

            if not image_dir.is_dir():
                continue

            image_paths = sorted(
                image_dir.glob("slice-*.png"),
                key=lambda p: int(
                    p.stem.split("-")[-1]
                ),
            )

            if len(image_paths) < 2:
                continue

            mask_dir = (
                MASK_ROOT
                / patient_dir.name
                / nodule_dir.name
            )

            # Require a matching consensus mask
            # for every image in the stack.
            if not all(
                (mask_dir / p.name).is_file()
                for p in image_paths
            ):
                continue

            records.append({
                "patient_id": patient_dir.name,
                "nodule_id": nodule_dir.name,
                "num_slices": len(image_paths),
            })

    return pd.DataFrame(records)


# ==================================================
# PATIENT-LEVEL SPLITTING
# ==================================================

def main():

    random.seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = build_metadata()

    if df.empty:
        raise RuntimeError(
            "No valid nodule stacks found."
        )

    patients = sorted(df["patient_id"].unique())

    # 80% train, 20% temporary
    train_patients, temp_patients = train_test_split(
        patients,
        test_size=0.20,
        random_state=SEED,
    )

    # Split remaining 20% into 10% validation
    # and 10% test.
    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=0.50,
        random_state=SEED,
    )

    splits = {
        "train": sorted(train_patients),
        "val": sorted(val_patients),
        "test": sorted(test_patients),
    }

    # Assign each nodule based on its patient.
    for split_name, patient_ids in splits.items():

        split_df = df[
            df["patient_id"].isin(patient_ids)
        ].copy()

        output_path = (
            OUTPUT_DIR
            / f"{split_name}_3d.csv"
        )

        split_df.to_csv(
            output_path,
            index=False,
        )

        print(
            f"{split_name.upper()}: "
            f"{len(patient_ids)} patients, "
            f"{len(split_df)} nodules"
        )

        print(f"Saved: {output_path}")

    # Save exact patient assignments for reproducibility.
    with open(
        OUTPUT_DIR / "patient_splits.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            splits,
            f,
            indent=2,
        )

    # Confirm that patient groups do not overlap.
    assert not (
        set(splits["train"])
        & set(splits["val"])
    )

    assert not (
        set(splits["train"])
        & set(splits["test"])
    )

    assert not (
        set(splits["val"])
        & set(splits["test"])
    )

    print("\nPatient-level split validation passed.")
    print("Total patients:", len(patients))
    print("Total nodules:", len(df))
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()