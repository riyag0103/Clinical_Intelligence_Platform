import sys
import os

# ----------------------------------
# ADD PROJECT ROOT
# ----------------------------------
project_root = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if project_root not in sys.path:
    sys.path.append(project_root)

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import glob

from utils.postprocess import postprocess_mask

# ----------------------------------
# IMPORT ATTENTION U-NET
# ----------------------------------
from models.attention_unet import AttentionUNet

# ----------------------------------
# DEVICE
# ----------------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"\nUSING DEVICE: {device}")

# ----------------------------------
# LOAD MODEL
# ----------------------------------
model = AttentionUNet().to(device)

# ----------------------------------
# LOAD TRAINED WEIGHTS
# ----------------------------------
ckpt_path = "outputs/checkpoints/best_attention_unet.pth"
if os.path.exists(ckpt_path):
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device)
    )
else:
    print(f"Warning: checkpoint not found at {ckpt_path}. Running with random weights.")

model.eval()

print("\nATTENTION U-NET LOADED SUCCESSFULLY!")

# ----------------------------------
# IMAGE / MASK PATHS
# ----------------------------------

def _find_first_image():
    patterns = [
        os.path.join("datasets", "processed", "enhanced_ct", "**", "images", "*.png"),
        os.path.join("datasets", "processed", "ct", "**", "images", "*.png"),
        os.path.join("datasets", "processed", "normalized_ct", "**", "images", "*.png"),
        os.path.join("datasets", "processed", "enhanced_ct", "**", "*.png"),
        os.path.join("datasets", "processed", "ct", "**", "*.png"),
    ]
    for p in patterns:
        found = glob.glob(p, recursive=True)
        if found:
            return found[0]
    return None


# Default hardcoded paths (may not exist)
image_path = r"datasets/processed/ct/images/slice_0.png"
mask_path = r"datasets/processed/ct/masks/slice_0.png"

# If hardcoded image missing, try to locate one automatically
if not os.path.exists(image_path):
    alt = _find_first_image()
    if alt:
        print(f"Hardcoded image not found, using: {alt}")
        image_path = alt
    else:
        raise FileNotFoundError("No image files found under datasets/processed/. Set `image_path` manually.")

# Infer mask path from image path when possible
def _find_corresponding_mask(img_path):
    # naive replace: enhanced_ct/.../images/... -> masks/... (or look under processed/masks)
    masks_dir = os.path.join("datasets", "processed", "masks")
    candidates = glob.glob(os.path.join(masks_dir, "**", "*.png"), recursive=True)
    if not candidates:
        return None
    # Try to match patient id in path
    for c in candidates:
        if os.path.basename(os.path.dirname(os.path.dirname(c))) in img_path:
            return c
    return candidates[0]

if not os.path.exists(mask_path):
    inferred = _find_corresponding_mask(image_path)
    if inferred:
        print(f"Inferred mask path: {inferred}")
        mask_path = inferred
    else:
        raise FileNotFoundError("No mask files found under datasets/processed/masks/. Set `mask_path` manually.")

# ----------------------------------
# LOAD IMAGE
# ----------------------------------
image = cv2.imread(
    image_path,
    cv2.IMREAD_GRAYSCALE
)

# ----------------------------------
# LOAD TRUE MASK
# ----------------------------------
true_mask = cv2.imread(
    mask_path,
    cv2.IMREAD_GRAYSCALE
)

# ----------------------------------
# CHECK FILES
# ----------------------------------
if image is None:

    raise FileNotFoundError(
        f"Image not found:\n{image_path}"
    )

if true_mask is None:

    raise FileNotFoundError(
        f"Mask not found:\n{mask_path}"
    )

# ----------------------------------
# NORMALIZE IMAGE
# ----------------------------------
image_normalized = image.astype(
    np.float32
) / 255.0

# ----------------------------------
# CREATE INPUT TENSOR
# ----------------------------------
input_tensor = torch.tensor(
    image_normalized
).unsqueeze(0).unsqueeze(0)

input_tensor = input_tensor.to(device)

# ----------------------------------
# MODEL INFERENCE
# ----------------------------------
with torch.no_grad():

    prediction = model(input_tensor)

# ----------------------------------
# CONVERT PREDICTION
# ----------------------------------
prediction = prediction.squeeze().cpu().numpy()

# ----------------------------------
# THRESHOLD PREDICTION
# ----------------------------------
prediction_mask = (
    prediction > 0.35
).astype(np.uint8)

prediction_mask = postprocess_mask(
    prediction_mask
)
# ----------------------------------
# CREATE OVERLAY
# ----------------------------------
overlay = cv2.cvtColor(
    image,
    cv2.COLOR_GRAY2BGR
)

overlay[
    prediction_mask == 1
] = [255, 0, 0]

# ----------------------------------
# VISUALIZATION
# ----------------------------------
plt.figure(figsize=(20, 5))

# ----------------------------------
# ORIGINAL CT
# ----------------------------------
plt.subplot(1, 5, 1)

plt.imshow(image, cmap="gray")

plt.title("Original CT Scan")

plt.axis("off")

# ----------------------------------
# TRUE MASK
# ----------------------------------
plt.subplot(1, 5, 2)

plt.imshow(true_mask, cmap="gray")

plt.title("Ground Truth Mask")

plt.axis("off")

# ----------------------------------
# PROBABILITY MAP
# ----------------------------------
plt.subplot(1, 5, 3)

plt.imshow(prediction, cmap="hot")

plt.title("Prediction Probability Map")

plt.axis("off")

# ----------------------------------
# PREDICTED MASK
# ----------------------------------
plt.subplot(1, 5, 4)

plt.imshow(prediction_mask, cmap="gray")

plt.title("Predicted Tumor Mask")

plt.axis("off")

# ----------------------------------
# OVERLAY
# ----------------------------------
plt.subplot(1, 5, 5)

plt.imshow(overlay)

plt.title("Tumor Overlay")

plt.axis("off")

plt.tight_layout()

# Ensure outputs directory exists and save result instead of opening an interactive window
os.makedirs("outputs", exist_ok=True)
out_file = os.path.join("outputs", "inference.png")
plt.savefig(out_file, bbox_inches='tight')
print(f"Inference visualization saved to: {out_file}")
plt.close()