from pathlib import Path

import cv2
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[2]

# -----------------------------
# IMAGE PATH
# -----------------------------
image_path = BASE_DIR / "datasets/raw/ct/LIDC-IDRI-slices/LIDC-IDRI-0001/nodule-0/images/slice-0.png"

# -----------------------------
# MASK PATH
# -----------------------------
mask_path = BASE_DIR / "datasets/raw/ct/LIDC-IDRI-slices/LIDC-IDRI-0001/nodule-0/mask-0/slice-0.png"

# -----------------------------
# LOAD IMAGE
# -----------------------------
image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

# -----------------------------
# LOAD MASK
# -----------------------------
mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

if image is None:
	raise FileNotFoundError(f"Could not load CT image: {image_path}")

if mask is None:
	raise FileNotFoundError(f"Could not load CT mask: {mask_path}")

# -----------------------------
# DISPLAY
# -----------------------------
plt.figure(figsize=(10,5))

# CT IMAGE
plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.title("CT Scan")
plt.axis("off")

# MASK
plt.subplot(1,2,2)
plt.imshow(mask, cmap='gray')
plt.title("Tumor Mask")
plt.axis("off")

plt.show()