import cv2
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# PATHS
# -----------------------------
image_path = r"datasets/raw/ct/LIDC-IDRI-slices/LIDC-IDRI-0001/nodule-0/images/slice-0.png"

mask_path = r"datasets/raw/ct/LIDC-IDRI-slices/LIDC-IDRI-0001/nodule-0/mask-0/slice-0.png"

# -----------------------------
# LOAD IMAGE
# -----------------------------
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# -----------------------------
# LOAD MASK
# -----------------------------
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# -----------------------------
# CONVERT IMAGE TO RGB
# -----------------------------
image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

# -----------------------------
# CREATE RED OVERLAY
# -----------------------------
overlay = image_rgb.copy()

overlay[mask > 0] = [255, 0, 0]

# -----------------------------
# BLEND IMAGE + MASK
# -----------------------------
blended = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)

# -----------------------------
# DISPLAY
# -----------------------------
plt.figure(figsize=(8,8))

plt.imshow(blended)

plt.title("Tumor Overlay Visualization")

plt.axis("off")

plt.show()