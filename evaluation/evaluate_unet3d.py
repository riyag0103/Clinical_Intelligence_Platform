import sys
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TEST_CSV = (
    PROJECT_ROOT
    / "datasets"
    / "metadata"
    / "3d_splits"
    / "test_3d.csv"
)

IMAGE_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "ct"
)

MASK_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "lidc_consensus_masks"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "unet3d"
    / "best_unet3d.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "unet3d"
    / "test_evaluation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = OUTPUT_DIR / "test_metrics.csv"
SUMMARY_FILE = OUTPUT_DIR / "test_summary.txt"

BATCH_SIZE = 1
THRESHOLD = 0.5
NUM_WORKERS = 0

# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 65)
print("3D U-NET INDEPENDENT TEST EVALUATION")
print("=" * 65)

print(f"Project root: {PROJECT_ROOT}")
print(f"Device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# 3. CHECK REQUIRED FILES
# ============================================================

if not TEST_CSV.exists():
    raise FileNotFoundError(
        f"Test CSV not found:\n{TEST_CSV}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found:\n{MODEL_PATH}"
    )

# ============================================================
# 4. LOAD MODEL AND TEST DATASET
# ============================================================

from models.unet3d import UNet3D
from data_loader.dataset_3d_patches import CT3DPatchDataset

print("\nLoading trained model...")

model = UNet3D().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

# Supports checkpoints saved as a dictionary or a raw state dict.
if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

print("Model loaded successfully.")

if isinstance(checkpoint, dict) and "epoch" in checkpoint:
    print(f"Checkpoint epoch: {checkpoint['epoch']}")

print("\nLoading test dataset...")

test_dataset = CT3DPatchDataset(
    csv_file=str(TEST_CSV),
    image_root=str(IMAGE_ROOT),
    mask_root=str(MASK_ROOT),
    patch_depth=8,
    stride=4
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda")
)

print(f"Test patches: {len(test_dataset)}")

# ============================================================
# 5. EVALUATION
# ============================================================

intersection_total = 0
prediction_total = 0
target_total = 0

true_positive = 0
false_positive = 0
false_negative = 0
true_negative = 0

patch_results = []

use_amp = device.type == "cuda"

print("\nEvaluating test patches...")

with torch.inference_mode():

    for index, batch in enumerate(test_loader, start=1):

        images = batch["image"].to(
            device,
            non_blocking=True
        )

        masks = batch["mask"].to(
            device,
            non_blocking=True
        )

        valid_slices = batch["valid_slices"].to(
            device,
            non_blocking=True
        )

        with autocast(
            device_type="cuda",
            enabled=use_amp
        ):
            logits = model(images)

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= THRESHOLD
        ).to(torch.uint8)

        targets = (
            masks >= 0.5
        ).to(torch.uint8)

        # Exclude artificial padded slices.
        valid_mask = valid_slices[
            :, None, :, None, None
        ].to(
            device=device,
            dtype=torch.bool
        )

        predictions = predictions[valid_mask.expand_as(predictions)]
        targets = targets[valid_mask.expand_as(targets)]

        intersection = int(
            (predictions * targets).sum().item()
        )

        predicted_positive = int(
            predictions.sum().item()
        )

        target_positive = int(
            targets.sum().item()
        )

        tp = intersection
        fp = predicted_positive - tp
        fn = target_positive - tp

        # TN is calculated over valid voxels.
        valid_voxels = int(valid_mask.expand_as(
            torch.sigmoid(logits)
        ).sum().item())

        tn = valid_voxels - tp - fp - fn

        intersection_total += intersection
        prediction_total += predicted_positive
        target_total += target_positive

        true_positive += tp
        false_positive += fp
        false_negative += fn
        true_negative += tn

        patch_dice = (
            2 * intersection
            / (predicted_positive + target_positive)
            if predicted_positive + target_positive > 0
            else 1.0
        )

        patch_iou = (
            intersection
            / (predicted_positive + target_positive - intersection)
            if predicted_positive + target_positive - intersection > 0
            else 1.0
        )

        patch_results.append({
            "patch_index": index,
            "dice": patch_dice,
            "iou": patch_iou,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn
        })

        if index % 50 == 0 or index == len(test_loader):
            print(
                f"Processed [{index}/{len(test_loader)}] test patches"
            )

# ============================================================
# 6. CALCULATE AGGREGATE METRICS
# ============================================================

dice = (
    2 * intersection_total
    / (prediction_total + target_total)
    if prediction_total + target_total > 0
    else 1.0
)

iou = (
    intersection_total
    / (prediction_total + target_total - intersection_total)
    if prediction_total + target_total - intersection_total > 0
    else 1.0
)

precision = (
    true_positive / (true_positive + false_positive)
    if true_positive + false_positive > 0
    else 0.0
)

recall = (
    true_positive / (true_positive + false_negative)
    if true_positive + false_negative > 0
    else 0.0
)

specificity = (
    true_negative / (true_negative + false_positive)
    if true_negative + false_positive > 0
    else 0.0
)

# This is voxel-level accuracy, not patient-level accuracy.
accuracy = (
    (true_positive + true_negative)
    / (
        true_positive
        + true_negative
        + false_positive
        + false_negative
    )
    if (
        true_positive
        + true_negative
        + false_positive
        + false_negative
    ) > 0
    else 0.0
)

# ============================================================
# 7. SAVE PER-PATCH RESULTS
# ============================================================

with open(
    RESULTS_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "patch_index",
        "dice",
        "iou",
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(patch_results)

# ============================================================
# 8. SAVE SUMMARY
# ============================================================

summary = f"""
3D U-NET TEST EVALUATION
========================

Checkpoint: {MODEL_PATH}
Test dataset: {TEST_CSV}
Threshold: {THRESHOLD}
Test patches evaluated: {len(patch_results)}

Aggregate voxel-level metrics
-----------------------------
Dice Score:  {dice:.4f}
IoU Score:   {iou:.4f}
Precision:   {precision:.4f}
Recall:      {recall:.4f}
Specificity: {specificity:.4f}
Accuracy:    {accuracy:.4f}

Confusion counts
----------------
True Positive:  {true_positive}
False Positive: {false_positive}
False Negative: {false_negative}
True Negative:  {true_negative}

IMPORTANT
---------
These metrics aggregate predictions across patches.
Overlapping patches can count the same slice more than once.
They are preliminary metrics, not final per-nodule results
or evidence of clinical performance.
"""

SUMMARY_FILE.write_text(
    summary.strip() + "\n",
    encoding="utf-8"
)

# ============================================================
# 9. PRINT RESULTS
# ============================================================

print("\n" + "=" * 65)
print("TEST EVALUATION COMPLETE")
print("=" * 65)

print(f"Dice Score:  {dice:.4f}")
print(f"IoU Score:   {iou:.4f}")
print(f"Precision:   {precision:.4f}")
print(f"Recall:      {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Accuracy:    {accuracy:.4f}")

print("\nResults saved to:")
print(RESULTS_CSV)
print(SUMMARY_FILE)