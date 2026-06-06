import pandas as pd
import cv2
import os

# ----------------------------------
# LOAD METADATA CSV
# ----------------------------------
csv_path = r"datasets/metadata/ct_metadata.csv"

df = pd.read_csv(csv_path)

# ----------------------------------
# VALIDATION COUNTERS
# ----------------------------------
missing_images = 0
missing_masks = 0
corrupted_images = 0
empty_masks = 0

# ----------------------------------
# VALIDATE DATASET
# ----------------------------------
for index, row in df.iterrows():

    image_path = row["image_path"]
    mask_path = row["mask_path"]

    # ----------------------------------
    # CHECK IMAGE EXISTS
    # ----------------------------------
    if not os.path.exists(image_path):
        missing_images += 1
        continue

    # ----------------------------------
    # CHECK MASK EXISTS
    # ----------------------------------
    if not os.path.exists(mask_path):
        missing_masks += 1
        continue

    # ----------------------------------
    # LOAD IMAGE
    # ----------------------------------
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # ----------------------------------
    # CHECK CORRUPTED IMAGE
    # ----------------------------------
    if image is None:
        corrupted_images += 1
        continue

    # ----------------------------------
    # LOAD MASK
    # ----------------------------------
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # ----------------------------------
    # CHECK EMPTY MASK
    # ----------------------------------
    if mask is None or mask.sum() == 0:
        empty_masks += 1

# ----------------------------------
# PRINT RESULTS
# ----------------------------------
print("\\nDATASET VALIDATION REPORT")
print("-" * 40)

print(f"Total records: {len(df)}")
print(f"Missing images: {missing_images}")
print(f"Missing masks: {missing_masks}")
print(f"Corrupted images: {corrupted_images}")
print(f"Empty masks: {empty_masks}")

print("\\nValidation completed successfully!")