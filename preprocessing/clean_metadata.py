import pandas as pd
import cv2
import os

# ----------------------------------
# LOAD ORIGINAL CSV
# ----------------------------------
csv_path = r"datasets/metadata/ct_metadata.csv"

df = pd.read_csv(csv_path)

# ----------------------------------
# CLEAN RECORD STORAGE
# ----------------------------------
clean_records = []

# ----------------------------------
# VALIDATE EACH RECORD
# ----------------------------------
for index, row in df.iterrows():

    image_path = row["image_path"]
    mask_path = row["mask_path"]

    # ----------------------------------
    # CHECK IMAGE EXISTS
    # ----------------------------------
    if not os.path.exists(image_path):
        continue

    # ----------------------------------
    # CHECK MASK EXISTS
    # ----------------------------------
    if not os.path.exists(mask_path):
        continue

    # ----------------------------------
    # LOAD MASK
    # ----------------------------------
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Skip invalid masks
    if mask is None:
        continue

    # ----------------------------------
    # SKIP EMPTY MASKS
    # ----------------------------------
    if mask.sum() == 0:
        continue

    # ----------------------------------
    # KEEP VALID RECORD
    # ----------------------------------
    clean_records.append(row)

# ----------------------------------
# CREATE CLEAN DATAFRAME
# ----------------------------------
clean_df = pd.DataFrame(clean_records)

# ----------------------------------
# SAVE CLEAN CSV
# ----------------------------------
output_csv = r"datasets/metadata/clean_ct_metadata.csv"

clean_df.to_csv(output_csv, index=False)

# ----------------------------------
# PRINT RESULTS
# ----------------------------------
print("\\nCLEAN DATASET GENERATED")
print("-" * 40)

print(f"Original records: {len(df)}")
print(f"Clean records: {len(clean_df)}")

print("\\nClean metadata saved successfully!")