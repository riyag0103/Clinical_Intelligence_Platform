import cv2
import os
import numpy as np
from tqdm import tqdm

# ----------------------------------
# INPUT FOLDER
# ----------------------------------
input_dir = r"datasets/processed/ct"

# ----------------------------------
# OUTPUT FOLDER
# ----------------------------------
output_dir = r"datasets/processed/normalized_ct"

os.makedirs(output_dir, exist_ok=True)

# ----------------------------------
# PROCESS ALL IMAGES
# ----------------------------------
for root, dirs, files in os.walk(input_dir):

    for file in tqdm(files):

        if file.endswith(".png"):

            # Full image path
            input_path = os.path.join(root, file)

            # Load image
            image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

            # Skip corrupted images
            if image is None:
                continue

            # ----------------------------------
            # NORMALIZATION
            # ----------------------------------
            normalized_image = image / 255.0

            # Convert back to uint8 for saving
            normalized_image = (normalized_image * 255).astype(np.uint8)

            # ----------------------------------
            # CREATE OUTPUT PATH
            # ----------------------------------
            relative_path = os.path.relpath(root, input_dir)

            save_folder = os.path.join(output_dir, relative_path)

            os.makedirs(save_folder, exist_ok=True)

            output_path = os.path.join(save_folder, file)

            # Save image
            cv2.imwrite(output_path, normalized_image)

print("Normalization completed successfully!")