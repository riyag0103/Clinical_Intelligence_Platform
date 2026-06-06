import cv2
import os
from tqdm import tqdm

# ----------------------------------
# INPUT MASK DIRECTORY
# ----------------------------------
input_dir = r"datasets/raw/ct/LIDC-IDRI-slices"

# ----------------------------------
# OUTPUT MASK DIRECTORY
# ----------------------------------
output_dir = r"datasets/processed/masks"

os.makedirs(output_dir, exist_ok=True)

# ----------------------------------
# TARGET SIZE
# ----------------------------------
TARGET_SIZE = (256, 256)

# ----------------------------------
# PROCESS ALL MASKS
# ----------------------------------
for root, dirs, files in os.walk(input_dir):

    # Process only mask folders
    if "mask" in root.lower():

        for file in tqdm(files):

            if file.endswith(".png"):

                # Full input path
                input_path = os.path.join(root, file)

                # Load mask
                mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

                # Skip corrupted masks
                if mask is None:
                    continue

                # ----------------------------------
                # RESIZE MASK
                # IMPORTANT:
                # Use INTER_NEAREST
                # ----------------------------------
                resized_mask = cv2.resize(
                    mask,
                    TARGET_SIZE,
                    interpolation=cv2.INTER_NEAREST
                )

                # ----------------------------------
                # CREATE OUTPUT PATH
                # ----------------------------------
                relative_path = os.path.relpath(root, input_dir)

                save_folder = os.path.join(output_dir, relative_path)

                os.makedirs(save_folder, exist_ok=True)

                output_path = os.path.join(save_folder, file)

                # Save processed mask
                cv2.imwrite(output_path, resized_mask)

print("Mask processing completed successfully!")