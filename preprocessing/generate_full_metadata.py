import os
import glob
import cv2
import pandas as pd

# ----------------------------------
# IMAGE ROOT DIRECTORY
# ----------------------------------
images_root = r"datasets/processed/enhanced_ct"

# ----------------------------------
# MASK ROOT DIRECTORY
# ----------------------------------
masks_root = r"datasets/processed/masks"

# ----------------------------------
# OUTPUT DIRECTORY
# ----------------------------------
output_dir = r"datasets/metadata"

os.makedirs(
    output_dir,
    exist_ok=True
)

# ----------------------------------
# FIND ALL IMAGE SLICES (NESTED STRUCTURE)
# ----------------------------------
image_paths = glob.glob(
    os.path.join(
        images_root,
        "*",
        "nodule-*",
        "images",
        "*.png"
    )
)

print(f"\nFOUND {len(image_paths)} IMAGE SLICES")

# ----------------------------------
# METADATA STORAGE
# ----------------------------------
data = []

# ----------------------------------
# PROCESS EACH IMAGE
# ----------------------------------
for image_path in image_paths:

    # image_path example:
    # datasets/processed/enhanced_ct/LIDC-IDRI-0001/nodule-0/images/slice-0.png
    parts = os.path.normpath(image_path).split(os.sep)

    if len(parts) < 6:
        continue

    patient_id = parts[-4]
    nodule_id = parts[-3]
    filename = parts[-1]

    # Match all annotator masks for this image
    mask_pattern = os.path.join(
        masks_root,
        patient_id,
        nodule_id,
        "mask-*",
        filename
    )

    mask_paths = glob.glob(mask_pattern)

    if not mask_paths:
        continue

    for mask_path in mask_paths:

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            continue

        empty_mask = (
            mask.sum() == 0
        )

        data.append({
            "image_path": image_path,
            "mask_path": mask_path,
            "patient_id": patient_id,
            "nodule_id": nodule_id,
            "empty_mask": empty_mask
        })

# ----------------------------------
# CREATE DATAFRAME
# ----------------------------------
df = pd.DataFrame(
    data,
    columns=[
        "image_path",
        "mask_path",
        "patient_id",
        "nodule_id",
        "empty_mask"
    ]
)

# ----------------------------------
# SAVE CSV
# ----------------------------------
output_path = os.path.join(
    output_dir,
    "full_metadata.csv"
)

df.to_csv(
    output_path,
    index=False
)

# ----------------------------------
# REPORT
# ----------------------------------
print("\nFULL METADATA GENERATED!")

print(f"\nTOTAL SAMPLES: {len(df)}")

if len(df) > 0:
    print(f"\nEMPTY MASKS: {df['empty_mask'].sum()}")

    print(f"\nTUMOR SLICES: {(~df['empty_mask']).sum()}")
else:
    print("\nEMPTY MASKS: 0")

    print("\nTUMOR SLICES: 0")

print(f"\nSaved To:\n{output_path}")