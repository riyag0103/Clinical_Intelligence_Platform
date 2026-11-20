from pathlib import Path
from collections import defaultdict
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\Clinical_Project")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loader.dataset_3d_patches import CT3DPatchDataset
from models.unet3d import UNet3D


IMAGE_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "ct"
    / "LIDC-IDRI-slices"
)

MASK_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "lidc_consensus_masks"
)

SPLIT_CSV = (
    PROJECT_ROOT
    / "datasets"
    / "metadata"
    / "3d_splits"
    / "val_3d.csv"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "unet3d"
    / "best_unet3d.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "unet3d"
    / "nodule_evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

PATCH_DEPTH = 8
STRIDE = 4

BATCH_SIZE = 1
NUM_WORKERS = 0

# Initial threshold. Threshold selection should use validation
# predictions only, not repeatedly tune against the test set.
THRESHOLD = 0.50

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("3D U-NET NODULE-LEVEL EVALUATION")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Split CSV: {SPLIT_CSV}")
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Threshold: {THRESHOLD}")


# ============================================================
# 3. CHECK REQUIRED FILES
# ============================================================

required_paths = [
    IMAGE_ROOT,
    MASK_ROOT,
    SPLIT_CSV,
    CHECKPOINT_PATH,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(
            f"Required path does not exist:\n{path}"
        )


# ============================================================
# 4. CREATE DATASET
# ============================================================

dataset = CT3DPatchDataset(
    csv_file=str(SPLIT_CSV),
    image_root=str(IMAGE_ROOT),
    mask_root=str(MASK_ROOT),
    patch_depth=PATCH_DEPTH,
    stride=STRIDE,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)

print(f"Total patches: {len(dataset)}")


# ============================================================
# 5. LOAD EXISTING MODEL CHECKPOINT
# ============================================================

# This evaluation does not train or modify the checkpoint.

model = UNet3D().to(DEVICE)

try:
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )
except TypeError:
    # Compatibility with older PyTorch versions.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )


if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        # The checkpoint may itself be a state dictionary.
        state_dict = checkpoint

else:
    state_dict = checkpoint


model.load_state_dict(state_dict)
model.eval()

print("Checkpoint loaded successfully.")


# ============================================================
# 6. METRIC FUNCTIONS
# ============================================================

def calculate_metrics(
    target,
    prediction,
    epsilon=1e-7,
):
    """
    Calculates binary segmentation metrics for one nodule.

    Inputs must contain each original slice only once.
    """

    target = np.asarray(
        target,
        dtype=bool,
    )

    prediction = np.asarray(
        prediction,
        dtype=bool,
    )

    tp = int(
        np.logical_and(target, prediction).sum()
    )

    fp = int(
        np.logical_and(~target, prediction).sum()
    )

    fn = int(
        np.logical_and(target, ~prediction).sum()
    )

    tn = int(
        np.logical_and(~target, ~prediction).sum()
    )

    # Empty reference and empty prediction:
    # Dice and IoU are defined as 1 for this nodule.
    if target.sum() == 0 and prediction.sum() == 0:
        dice = 1.0
        iou = 1.0

    else:
        dice = (
            2.0 * tp
            / (2.0 * tp + fp + fn + epsilon)
        )

        iou = (
            tp
            / (tp + fp + fn + epsilon)
        )

    precision = (
        tp / (tp + fp + epsilon)
    )

    recall = (
        tp / (tp + fn + epsilon)
    )

    specificity = (
        tn / (tn + fp + epsilon)
    )

    accuracy = (
        (tp + tn)
        / (tp + tn + fp + fn + epsilon)
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "reference_pixels": int(target.sum()),
        "predicted_pixels": int(prediction.sum()),
    }


# ============================================================
# 7. COLLECT PATCH PREDICTIONS
# ============================================================

# Each nodule contains:
#   probabilities: accumulated prediction probabilities
#   counts: number of predictions covering each slice
#   targets: reference mask for each original slice
#
# Overlapping patches will be averaged before thresholding.

nodule_storage = {}

with torch.inference_mode():

    progress = tqdm(
        loader,
        desc="Evaluating patches",
    )

    for batch in progress:

        images = batch["image"].to(
            DEVICE,
            non_blocking=True,
        )

        masks = batch["mask"].cpu().numpy()

        valid_slices = batch[
            "valid_slices"
        ].cpu().numpy()

        start_indices = batch[
            "start_idx"
        ].cpu().numpy()

        real_depths = batch[
            "real_depth"
        ].cpu().numpy()

        patient_ids = batch["patient_id"]
        nodule_ids = batch["nodule_id"]

        logits = model(images)

        # Model must return raw logits.
        probabilities = torch.sigmoid(
            logits
        ).cpu().numpy()

        batch_size = images.shape[0]

        for b in range(batch_size):

            patient_id = str(
                patient_ids[b]
            )

            nodule_id = str(
                nodule_ids[b]
            )

            key = (
                patient_id,
                nodule_id,
            )

            start_idx = int(
                start_indices[b]
            )

            real_depth = int(
                real_depths[b]
            )

            valid_depth = int(
                valid_slices[b].sum()
            )

            if real_depth != valid_depth:
                raise RuntimeError(
                    "Mismatch between real_depth and "
                    "valid_slices for "
                    f"{key}, start={start_idx}"
                )

            patch_probabilities = probabilities[
                b, 0, :real_depth
            ]

            patch_masks = masks[
                b, 0, :real_depth
            ]

            if key not in nodule_storage:

                nodule_storage[key] = {
                    "probability_sum": {},
                    "prediction_count": {},
                    "targets": {},
                    "slice_names": {},
                }

            record = nodule_storage[key]

            # The dataset has ordered the image paths by slice
            # number, so start_idx maps directly to their
            # original position within this nodule.

            for local_idx in range(real_depth):

                original_idx = (
                    start_idx + local_idx
                )

                probability = patch_probabilities[
                    local_idx
                ]

                target_mask = (
                    patch_masks[local_idx] > 0.5
                )

                if original_idx not in record[
                    "probability_sum"
                ]:

                    record[
                        "probability_sum"
                    ][original_idx] = (
                        probability.astype(
                            np.float64
                        ).copy()
                    )

                    record[
                        "prediction_count"
                    ][original_idx] = 1

                    record[
                        "targets"
                    ][original_idx] = (
                        target_mask.copy()
                    )

                else:

                    existing_target = record[
                        "targets"
                    ][original_idx]

                    # The reference mask should be identical
                    # wherever a slice occurs in overlapping
                    # patches.
                    if not np.array_equal(
                        existing_target,
                        target_mask,
                    ):
                        raise RuntimeError(
                            "Reference mask mismatch across "
                            f"overlapping patches: {key}, "
                            f"slice index {original_idx}"
                        )

                    record[
                        "probability_sum"
                    ][original_idx] += (
                        probability
                    )

                    record[
                        "prediction_count"
                    ][original_idx] += 1


# ============================================================
# 8. RECONSTRUCT EACH NODULE
# ============================================================

nodule_results = []
slice_results = []

global_tp = 0
global_fp = 0
global_fn = 0
global_tn = 0

global_target_pixels = 0
global_prediction_pixels = 0

print("\nReconstructing nodules...")

for key, record in nodule_storage.items():

    patient_id, nodule_id = key

    slice_indices = sorted(
        record["probability_sum"].keys()
    )

    if not slice_indices:
        continue

    nodule_targets = []
    nodule_predictions = []

    for slice_idx in slice_indices:

        probability_sum = record[
            "probability_sum"
        ][slice_idx]

        count = record[
            "prediction_count"
        ][slice_idx]

        # Average probabilities across overlapping patches.
        averaged_probability = (
            probability_sum / count
        )

        target = record[
            "targets"
        ][slice_idx]

        prediction = (
            averaged_probability >= THRESHOLD
        )

        nodule_targets.append(target)
        nodule_predictions.append(prediction)

        slice_metrics = calculate_metrics(
            target,
            prediction,
        )

        slice_results.append({
            "patient_id": patient_id,
            "nodule_id": nodule_id,
            "slice_index": slice_idx,
            "threshold": THRESHOLD,
            "overlap_count": count,
            **slice_metrics,
        })

    # Stack unique slices: [D,H,W].
    nodule_target_volume = np.stack(
        nodule_targets,
        axis=0,
    )

    nodule_prediction_volume = np.stack(
        nodule_predictions,
        axis=0,
    )

    metrics = calculate_metrics(
        nodule_target_volume,
        nodule_prediction_volume,
    )

    nodule_results.append({
        "patient_id": patient_id,
        "nodule_id": nodule_id,
        "num_slices": len(slice_indices),
        "first_slice_index": min(slice_indices),
        "last_slice_index": max(slice_indices),
        "threshold": THRESHOLD,
        **metrics,
    })

    global_tp += metrics["tp"]
    global_fp += metrics["fp"]
    global_fn += metrics["fn"]
    global_tn += metrics["tn"]

    global_target_pixels += metrics[
        "reference_pixels"
    ]

    global_prediction_pixels += metrics[
        "predicted_pixels"
    ]


# ============================================================
# 9. SAVE PER-NODULE AND PER-SLICE RESULTS
# ============================================================

if not nodule_results:
    raise RuntimeError(
        "No nodules were evaluated. "
        "Check the validation CSV and dataset paths."
    )

nodule_df = pd.DataFrame(
    nodule_results
)

slice_df = pd.DataFrame(
    slice_results
)

nodule_csv = (
    OUTPUT_DIR
    / "per_nodule_metrics.csv"
)

slice_csv = (
    OUTPUT_DIR
    / "per_slice_metrics.csv"
)

nodule_df.to_csv(
    nodule_csv,
    index=False,
)

slice_df.to_csv(
    slice_csv,
    index=False,
)


# ============================================================
# 10. OVERALL VOXEL METRICS
# ============================================================

epsilon = 1e-7

overall_dice = (
    2 * global_tp
    / (
        2 * global_tp
        + global_fp
        + global_fn
        + epsilon
    )
)

overall_iou = (
    global_tp
    / (
        global_tp
        + global_fp
        + global_fn
        + epsilon
    )
)

overall_precision = (
    global_tp
    / (
        global_tp
        + global_fp
        + epsilon
    )
)

overall_recall = (
    global_tp
    / (
        global_tp
        + global_fn
        + epsilon
    )
)

overall_specificity = (
    global_tn
    / (
        global_tn
        + global_fp
        + epsilon
    )
)

overall_accuracy = (
    (global_tp + global_tn)
    / (
        global_tp
        + global_fp
        + global_fn
        + global_tn
        + epsilon
    )
)


# ============================================================
# 11. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("NODULE-LEVEL EVALUATION SUMMARY")
print("=" * 70)

print(f"Evaluated nodules: {len(nodule_df)}")
print(f"Evaluated unique slices: {len(slice_df)}")
print(f"Probability threshold: {THRESHOLD:.2f}")

print("\nOVERALL VOXEL METRICS")
print("-" * 40)

print(f"Dice:        {overall_dice:.4f}")
print(f"IoU:         {overall_iou:.4f}")
print(f"Precision:   {overall_precision:.4f}")
print(f"Recall:      {overall_recall:.4f}")
print(f"Specificity: {overall_specificity:.4f}")
print(f"Accuracy:    {overall_accuracy:.4f}")

print("\nPER-NODULE METRICS")
print("-" * 40)

for metric in [
    "dice",
    "iou",
    "precision",
    "recall",
    "specificity",
    "accuracy",
]:

    values = nodule_df[metric]

    print(
        f"{metric.capitalize():12s} "
        f"Mean={values.mean():.4f} | "
        f"Median={values.median():.4f}"
    )

print("\nNODULES WITH EMPTY REFERENCE MASKS")
print("-" * 40)

empty_reference_count = int(
    (
        nodule_df["reference_pixels"] == 0
    ).sum()
)

print(
    f"{empty_reference_count} / "
    f"{len(nodule_df)}"
)

print("\nOUTPUT FILES")
print("-" * 40)

print(f"Per-nodule CSV: {nodule_csv}")
print(f"Per-slice CSV:  {slice_csv}")

print("\nEvaluation completed successfully.")