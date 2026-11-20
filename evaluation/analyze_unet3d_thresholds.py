import sys
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VAL_CSV = (
    PROJECT_ROOT
    / "datasets"
    / "metadata"
    / "3d_splits"
    / "val_3d.csv"
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
    / "threshold_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = OUTPUT_DIR / "threshold_metrics.csv"
GRAPH_PATH = OUTPUT_DIR / "threshold_comparison.png"

THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

BATCH_SIZE = 1
NUM_WORKERS = 0

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

use_amp = device.type == "cuda"


# ============================================================
# 2. CHECK FILES
# ============================================================

print("=" * 65)
print("3D U-NET VALIDATION THRESHOLD ANALYSIS")
print("=" * 65)

print(f"Device: {device}")

if not VAL_CSV.exists():
    raise FileNotFoundError(
        f"Validation CSV not found:\n{VAL_CSV}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found:\n{MODEL_PATH}"
    )


# ============================================================
# 3. LOAD MODEL
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

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model.eval()

print("Model loaded successfully.")


# ============================================================
# 4. LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation dataset...")

dataset = CT3DPatchDataset(
    csv_file=str(VAL_CSV),
    image_root=str(IMAGE_ROOT),
    mask_root=str(MASK_ROOT),
    patch_depth=8,
    stride=4
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

print(f"Validation patches: {len(dataset)}")


# ============================================================
# 5. COLLECT PREDICTIONS ONCE
# ============================================================

print("\nGenerating validation predictions...")

all_probabilities = []
all_targets = []

with torch.inference_mode():

    for batch_index, batch in enumerate(loader, start=1):

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

        # Exclude padded slices from metric calculations.
        valid_mask = valid_slices[
            :, None, :, None, None
        ].to(
            device=device,
            dtype=torch.bool
        )

        # Keep only valid voxels.
        probabilities = probabilities[valid_mask.expand_as(probabilities)]
        targets = masks[valid_mask.expand_as(masks)]

        all_probabilities.append(
            probabilities.float().cpu().numpy().reshape(-1)
        )

        all_targets.append(
            (targets >= 0.5).cpu().numpy().astype(np.uint8).reshape(-1)
        )

        if batch_index % 50 == 0 or batch_index == len(loader):
            print(
                f"Processed [{batch_index}/{len(loader)}] "
                "validation patches"
            )

if not all_probabilities:
    raise RuntimeError("No validation predictions were generated.")

all_probabilities = np.concatenate(all_probabilities)
all_targets = np.concatenate(all_targets).astype(bool)

if len(all_probabilities) != len(all_targets):
    raise RuntimeError(
        "Prediction and target sizes do not match."
    )

print(f"\nEvaluated voxels: {len(all_targets):,}")
print(f"Reference-positive voxels: {int(all_targets.sum()):,}")


# ============================================================
# 6. EVALUATE EACH THRESHOLD
# ============================================================

results = []

print("\n" + "=" * 65)
print("THRESHOLD RESULTS")
print("=" * 65)

print(
    f"{'Threshold':<12}"
    f"{'Dice':<12}"
    f"{'IoU':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
)

for threshold in THRESHOLDS:

    predictions = all_probabilities >= threshold

    tp = int(np.logical_and(predictions, all_targets).sum())
    fp = int(np.logical_and(predictions, ~all_targets).sum())
    fn = int(np.logical_and(~predictions, all_targets).sum())
    tn = int(np.logical_and(~predictions, ~all_targets).sum())

    dice_denominator = 2 * tp + fp + fn

    dice = (
        2 * tp / dice_denominator
        if dice_denominator > 0
        else 1.0
    )

    iou_denominator = tp + fp + fn

    iou = (
        tp / iou_denominator
        if iou_denominator > 0
        else 1.0
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
        else 0.0
    )

    results.append({
        "threshold": threshold,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn
    })

    print(
        f"{threshold:<12.2f}"
        f"{dice:<12.4f}"
        f"{iou:<12.4f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
    )


# ============================================================
# 7. SAVE RESULTS TO CSV
# ============================================================

fieldnames = list(results[0].keys())

with open(
    RESULTS_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)


# ============================================================
# 8. IDENTIFY BEST VALIDATION DICE THRESHOLD
# ============================================================

best_result = max(
    results,
    key=lambda row: row["dice"]
)

print("\n" + "=" * 65)
print("BEST VALIDATION THRESHOLD")
print("=" * 65)

print(f"Threshold:  {best_result['threshold']:.2f}")
print(f"Dice:       {best_result['dice']:.4f}")
print(f"IoU:        {best_result['iou']:.4f}")
print(f"Precision:  {best_result['precision']:.4f}")
print(f"Recall:     {best_result['recall']:.4f}")
print(f"Specificity:{best_result['specificity']:.4f}")


# ============================================================
# 9. PLOT METRICS AGAINST THRESHOLD
# ============================================================

threshold_values = [
    row["threshold"] for row in results
]

plt.figure(figsize=(10, 6))

plt.plot(
    threshold_values,
    [row["dice"] for row in results],
    marker="o",
    label="Dice"
)

plt.plot(
    threshold_values,
    [row["iou"] for row in results],
    marker="s",
    label="IoU"
)

plt.plot(
    threshold_values,
    [row["precision"] for row in results],
    marker="^",
    label="Precision"
)

plt.plot(
    threshold_values,
    [row["recall"] for row in results],
    marker="d",
    label="Recall"
)

plt.xlabel("Prediction Threshold")
plt.ylabel("Metric Score")
plt.title("3D U-Net Validation Threshold Analysis")

plt.xticks(threshold_values)
plt.ylim(0, 1.05)

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    GRAPH_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 10. FINISH
# ============================================================

print("\n" + "=" * 65)
print("THRESHOLD ANALYSIS COMPLETE")
print("=" * 65)

print(f"Results CSV:\n{RESULTS_CSV}")
print(f"Comparison graph:\n{GRAPH_PATH}")

print(
    "\nNote: Results aggregate predictions across patches. "
    "Overlapping patches can count some voxels more than once. "
    "Use these findings for preliminary threshold selection; "
    "final evaluation should reconstruct each nodule without "
    "duplicate counting."
)