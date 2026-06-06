import os
import pandas as pd

# ----------------------------------
# IMAGE DIRECTORY
# ----------------------------------
image_root = r"datasets/processed/enhanced_ct"

# ----------------------------------
# MASK DIRECTORY
# ----------------------------------
mask_root = r"datasets/processed/masks"

# ----------------------------------
# METADATA STORAGE
# ----------------------------------
metadata = []

# ----------------------------------
# WALK THROUGH IMAGE DATASET
# ----------------------------------
for root, dirs, files in os.walk(image_root):

    for file in files:

        if file.endswith(".png"):

            # ----------------------------------
            # IMAGE PATH
            # ----------------------------------
            image_path = os.path.join(root, file)

            # ----------------------------------
            # RELATIVE PATH
            # ----------------------------------
            relative_path = os.path.relpath(root, image_root)

            # ----------------------------------
            # MASK PATH
            # ----------------------------------
            mask_path = os.path.join(
                mask_root,
                relative_path,
                file
            )

            # ----------------------------------
            # EXTRACT INFORMATION
            # ----------------------------------
            path_parts = relative_path.split(os.sep)

            patient_id = "unknown"
            nodule_id = "unknown"

            for part in path_parts:

                if "LIDC-IDRI" in part:
                    patient_id = part

                if "nodule" in part:
                    nodule_id = part

            # ----------------------------------
            # LABEL
            # ----------------------------------
            label = "tumor"

            # ----------------------------------
            # STORE RECORD
            # ----------------------------------
            metadata.append({
                "patient_id": patient_id,
                "nodule_id": nodule_id,
                "modality": "CT",
                "image_path": image_path,
                "mask_path": mask_path,
                "label": label
            })

# ----------------------------------
# CREATE DATAFRAME
# ----------------------------------
df = pd.DataFrame(metadata)

# ----------------------------------
# SAVE CSV
# ----------------------------------
output_csv = r"datasets/metadata/ct_metadata.csv"

df.to_csv(output_csv, index=False)

print("Metadata CSV generated successfully!")
print(f"Total records: {len(df)}")