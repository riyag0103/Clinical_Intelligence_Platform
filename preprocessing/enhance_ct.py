import cv2
import os
from tqdm import tqdm

# ----------------------------------
# INPUT FOLDER
# ----------------------------------
input_dir = r"datasets/processed/normalized_ct"

# ----------------------------------
# OUTPUT FOLDER
# ----------------------------------
output_dir = r"datasets/processed/enhanced_ct"

os.makedirs(output_dir, exist_ok=True)

# ----------------------------------
# CREATE CLAHE OBJECT
# ----------------------------------
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8,8)
)

# ----------------------------------
# PROCESS ALL IMAGES
# ----------------------------------
for root, dirs, files in os.walk(input_dir):

    for file in tqdm(files):

        if file.endswith(".png"):

            # Full path
            input_path = os.path.join(root, file)

            # Load image
            image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

            # Skip corrupted images
            if image is None:
                continue

            # ----------------------------------
            # APPLY CLAHE
            # ----------------------------------
            enhanced_image = clahe.apply(image)

            # ----------------------------------
            # CREATE OUTPUT PATH
            # ----------------------------------
            relative_path = os.path.relpath(root, input_dir)

            save_folder = os.path.join(output_dir, relative_path)

            os.makedirs(save_folder, exist_ok=True)

            output_path = os.path.join(save_folder, file)

            # Save image
            cv2.imwrite(output_path, enhanced_image)

print("Contrast enhancement completed successfully!")