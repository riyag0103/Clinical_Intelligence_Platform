import sys
import os

# ----------------------------------
# ADD PROJECT ROOT
# ----------------------------------
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import cv2
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.attention_unet import AttentionUNet

from utils.metrics import (
    dice_score,
    iou_score
)

from utils.postprocess import (
    postprocess_mask
)

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

model.load_state_dict(
    torch.load(
        "outputs/checkpoints/best_attention_unet.pth",
        map_location=device
    )
)

model.eval()

print("\nMODEL LOADED SUCCESSFULLY!")

# ----------------------------------
# LOAD METADATA
# ----------------------------------
metadata_path = r"datasets/metadata/balanced/balanced_metadata.csv"

df = pd.read_csv(metadata_path)

# ----------------------------------
# LIMIT TEST SAMPLES
# ----------------------------------
NUM_SAMPLES = 10

df = df.sample(
    n=min(NUM_SAMPLES, len(df)),
    random_state=42
).reset_index(drop=True)

# ----------------------------------
# OUTPUT DIRECTORY
# ----------------------------------
output_dir = r"outputs/evaluation_results"

os.makedirs(
    output_dir,
    exist_ok=True
)

# ----------------------------------
# METRIC STORAGE
# ----------------------------------
all_dice = []
all_iou = []

# ----------------------------------
# EVALUATE EACH SAMPLE
# ----------------------------------
for idx, row in df.iterrows():

    image_path = row["image_path"]

    mask_path = row["mask_path"]

    # ----------------------------------
    # LOAD IMAGE
    # ----------------------------------
    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    # ----------------------------------
    # LOAD MASK
    # ----------------------------------
    true_mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None or true_mask is None:

        continue

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
    # INFERENCE
    # ----------------------------------
    with torch.no_grad():

        prediction = model(input_tensor)

    prediction = (
        prediction.squeeze()
        .cpu()
        .numpy()
    )

    # ----------------------------------
    # THRESHOLD
    # ----------------------------------
    prediction_mask = (
        prediction > 0.35
    ).astype(np.uint8)

    # ----------------------------------
    # POSTPROCESS
    # ----------------------------------
    prediction_mask = postprocess_mask(
        prediction_mask
    )

    # ----------------------------------
    # TRUE MASK BINARY
    # ----------------------------------
    true_mask_binary = (
        true_mask > 0
    ).astype(np.uint8)

    # ----------------------------------
    # CALCULATE METRICS
    # ----------------------------------
    pred_tensor = torch.tensor(
        prediction_mask
    ).float()

    true_tensor = torch.tensor(
        true_mask_binary
    ).float()

    dice = dice_score(
        pred_tensor,
        true_tensor
    )

    iou = iou_score(
        pred_tensor,
        true_tensor
    )

    all_dice.append(dice)

    all_iou.append(iou)

    # ----------------------------------
    # CREATE OVERLAY
    # ----------------------------------
    overlay = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR
    )

    overlay[
        prediction_mask == 1
    ] = [0, 0, 255]

    # ----------------------------------
    # VISUALIZATION
    # ----------------------------------
    plt.figure(figsize=(16, 4))

    # ORIGINAL
    plt.subplot(1, 4, 1)

    plt.imshow(image, cmap="gray")

    plt.title("CT Scan")

    plt.axis("off")

    # TRUE MASK
    plt.subplot(1, 4, 2)

    plt.imshow(true_mask_binary, cmap="gray")

    plt.title("Ground Truth")

    plt.axis("off")

    # PREDICTION
    plt.subplot(1, 4, 3)

    plt.imshow(prediction_mask, cmap="gray")

    plt.title("Prediction")

    plt.axis("off")

    # OVERLAY
    plt.subplot(1, 4, 4)

    plt.imshow(overlay)

    plt.title("Overlay")

    plt.axis("off")

    plt.tight_layout()

    # ----------------------------------
    # SAVE RESULT
    # ----------------------------------
    save_path = os.path.join(
        output_dir,
        f"result_{idx}.png"
    )

    plt.savefig(save_path)

    plt.close()

    print(f"\nProcessed Sample {idx+1}")

    print(f"Dice Score: {dice:.4f}")

    print(f"IoU Score: {iou:.4f}")

# ----------------------------------
# FINAL REPORT
# ----------------------------------
print("\n" + "=" * 50)

print("\nEVALUATION COMPLETED!")

print(f"\nAverage Dice Score: {np.mean(all_dice):.4f}")

print(f"\nAverage IoU Score: {np.mean(all_iou):.4f}")

print("\nResults saved in:")

print(output_dir)

print("\n" + "=" * 50)