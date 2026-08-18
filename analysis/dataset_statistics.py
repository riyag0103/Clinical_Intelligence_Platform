import os
import cv2
import pandas as pd
import numpy as np

# ----------------------------------
# LOAD METADATA
# ----------------------------------
metadata_path = r"datasets/metadata/balanced/balanced_metadata.csv"

df = pd.read_csv(metadata_path)

print("\nTOTAL SAMPLES:")
print(len(df))

# ----------------------------------
# COUNTERS
# ----------------------------------
tumor_images = 0
healthy_images = 0

tumor_areas = []

# ----------------------------------
# LOOP
# ----------------------------------
for idx, row in df.iterrows():

    mask_path = row["mask_path"]

    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        continue

    tumor_pixels = np.sum(mask > 0)

    total_pixels = (
        mask.shape[0]
        *
        mask.shape[1]
    )

    area_percent = (
        tumor_pixels / total_pixels
    ) * 100

    if tumor_pixels > 0:

        tumor_images += 1

        tumor_areas.append(
            area_percent
        )

    else:

        healthy_images += 1

# ----------------------------------
# REPORT
# ----------------------------------
print("\n" + "=" * 50)

print(
    f"Tumor Images: {tumor_images}"
)

print(
    f"Healthy Images: {healthy_images}"
)

if len(tumor_areas) > 0:

    print(
        f"\nAverage Tumor Area (%): "
        f"{np.mean(tumor_areas):.4f}"
    )

    print(
        f"Min Tumor Area (%): "
        f"{np.min(tumor_areas):.4f}"
    )

    print(
        f"Max Tumor Area (%): "
        f"{np.max(tumor_areas):.4f}"
    )

print("\n" + "=" * 50)