import cv2
import os
from tqdm import tqdm

# ----------------------------------
# INPUT DATASET PATH
# ----------------------------------
input_dir = r"datasets/raw/ct/LIDC-IDRI-slices"

# ----------------------------------
# OUTPUT DATASET PATH
# ----------------------------------
output_dir = r"datasets/processed/ct"

# Create output folder if not exists
os.makedirs(output_dir, exist_ok=True)

# ----------------------------------
# TARGET IMAGE SIZE
# ----------------------------------
TARGET_SIZE = (256, 256)

# ----------------------------------
# PROCESS ALL IMAGES
# ----------------------------------
for root, dirs, files in os.walk(input_dir):

    for file in tqdm(files):

        # Process only PNG images
        if file.endswith(".png"):

            # Full input path
            input_path = os.path.join(root, file)

            # Load image
            image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

            # Skip corrupted images
            if image is None:
                continue

            # Resize image
            resized_image = cv2.resize(image, TARGET_SIZE)

            # ----------------------------------
            # CREATE RELATIVE OUTPUT PATH
            # ----------------------------------
            relative_path = os.path.relpath(root, input_dir)

            save_folder = os.path.join(output_dir, relative_path)

            os.makedirs(save_folder, exist_ok=True)

            # Save image
            output_path = os.path.join(save_folder, file)

            cv2.imwrite(output_path, resized_image)

print("Dataset resizing completed successfully!")