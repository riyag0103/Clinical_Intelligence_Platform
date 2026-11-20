from pathlib import Path
import sys
import csv
import json
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

IMAGE_ROOT = PROJECT_ROOT / "datasets" / "processed" / "ct"
MASK_ROOT = PROJECT_ROOT / "datasets" / "processed" / "lidc_consensus_masks"
VAL_CSV = PROJECT_ROOT / "datasets" / "metadata" / "3d_splits" / "val_3d.csv"

ORIGINAL_CHECKPOINT = PROJECT_ROOT / "saved_models" / "unet3d" / "best_unet3d.pth"
DROPOUT_CHECKPOINT = PROJECT_ROOT / "saved_models" / "unet3d" / "best_unet3d_dropout.pth"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "unet3d" / "ensemble_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use the same patch settings as training/evaluation.
PATCH_DEPTH = 8
PATCH_STRIDE = 4
BATCH_SIZE = 1
NUM_WORKERS = 0
THRESHOLD = 0.50

# ---------------------------------------------------------------------
# Imports from your project
# ---------------------------------------------------------------------
try:
    from models.unet3d import UNet3D
    from data_loader.dataset_3d_patches import CT3DPatchDataset
except Exception as exc:
    raise ImportError(
        "Could not import UNet3D or CT3DPatchDataset. "
        "Check that models/unet3d.py and data_loader/dataset_3d_patches.py "
        "exist and that you are running this script from the project."
    ) from exc


def load_checkpoint(model, checkpoint_path, device):
    """Load either a wrapped training checkpoint or a plain state_dict."""
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def scalar(value):
    """Convert a collated tensor/list/scalar into a Python scalar."""
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0]
    if isinstance(value, (list, tuple)):
        value = value[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return value.item() if isinstance(value, np.generic) else value


def get_patch_array(tensor):
    """Convert [1, 1, D, H, W] or [1, D, H, W] tensor to [D, H, W]."""
    arr = tensor.detach().float().cpu().numpy()
    while arr.ndim > 3:
        arr = arr[0]
    return arr


def metric_counts(reference, prediction):
    reference = np.asarray(reference, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)

    tp = int(np.logical_and(reference, prediction).sum())
    fp = int(np.logical_and(~reference, prediction).sum())
    fn = int(np.logical_and(reference, ~prediction).sum())
    tn = int(np.logical_and(~reference, ~prediction).sum())

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "dice": (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 1.0,
        "iou": (tp / (tp + fp + fn)) if (tp + fp + fn) else 1.0,
        "precision": (tp / (tp + fp)) if (tp + fp) else 1.0,
        "recall": (tp / (tp + fn)) if (tp + fn) else 1.0,
        "specificity": (tn / (tn + fp)) if (tn + fp) else 1.0,
        "accuracy": ((tp + tn) / (tp + tn + fp + fn))
                    if (tp + tn + fp + fn) else 1.0,
        "reference_positive_voxels": int(reference.sum()),
        "predicted_positive_voxels": int(prediction.sum()),
    }


def add_patch(accumulator, key, start_idx, probabilities, reference):
    """
    Add a patch to a nodule accumulator. Overlapping patch probabilities
    are averaged slice-by-slice before thresholding.
    """
    if key not in accumulator:
        accumulator[key] = {"prob_sum": {}, "prob_count": {}, "reference": {}}

    item = accumulator[key]
    depth = min(probabilities.shape[0], reference.shape[0])

    for z in range(depth):
        slice_index = int(start_idx) + z
        prob_slice = probabilities[z]
        ref_slice = reference[z] > 0.5

        if slice_index not in item["prob_sum"]:
            item["prob_sum"][slice_index] = prob_slice.astype(np.float64, copy=True)
            item["prob_count"][slice_index] = 1
            item["reference"][slice_index] = ref_slice.copy()
        else:
            item["prob_sum"][slice_index] += prob_slice
            item["prob_count"][slice_index] += 1

            # Repeated patches for the same slice should carry the same mask.
            if not np.array_equal(item["reference"][slice_index], ref_slice):
                raise ValueError(
                    f"Mask mismatch for nodule {key}, slice position {slice_index}. "
                    "Check image/mask pairing and patch ordering."
                )


def evaluate():
    required_paths = [
        IMAGE_ROOT, MASK_ROOT, VAL_CSV,
        ORIGINAL_CHECKPOINT, DROPOUT_CHECKPOINT
    ]
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        print("\nERROR: These required paths were not found:")
        for path in missing:
            print("  -", path)
        print(
            "\nCheck the paths at the top of this script. "
            "IMAGE_ROOT must match the image path used during training."
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Both checkpoints are loaded into the current dropout-enabled class.
    # Dropout has no learned parameters and is disabled by model.eval().
    original_model = load_checkpoint(
        UNet3D(), ORIGINAL_CHECKPOINT, device
    )
    dropout_model = load_checkpoint(
        UNet3D(), DROPOUT_CHECKPOINT, device
    )

    dataset = CT3DPatchDataset(
        csv_file=str(VAL_CSV),
        image_root=str(IMAGE_ROOT),
        mask_root=str(MASK_ROOT),
        patch_depth=PATCH_DEPTH,
        stride=PATCH_STRIDE,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    # key -> accumulated probabilities/reference masks for each model
    accumulators = {
        "original": {},
        "dropout": {},
        "ensemble": {},
    }

    print(f"Validation patches: {len(dataset)}")
    print("Running inference and reconstructing nodule slices...")

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"]

            original_logits = original_model(images)
            dropout_logits = dropout_model(images)

            original_prob = torch.sigmoid(original_logits)
            dropout_prob = torch.sigmoid(dropout_logits)
            ensemble_prob = (original_prob + dropout_prob) / 2.0

            original_np = get_patch_array(original_prob)
            dropout_np = get_patch_array(dropout_prob)
            ensemble_np = get_patch_array(ensemble_prob)
            mask_np = get_patch_array(masks)

            patient_id = str(scalar(batch["patient_id"]))
            nodule_id = str(scalar(batch["nodule_id"]))
            start_idx = int(scalar(batch["start_idx"]))

            # real_depth is the number of genuine slices in this patch;
            # padded slices must not be included in reconstruction.
            if "real_depth" in batch:
                real_depth = int(scalar(batch["real_depth"]))
            else:
                warnings.warn(
                    "Dataset did not return real_depth; padded patch slices "
                    "may be included. Confirm your dataset implementation."
                )
                real_depth = PATCH_DEPTH

            real_depth = max(0, min(real_depth, PATCH_DEPTH))
            key = f"{patient_id}__{nodule_id}"

            add_patch(
                accumulators["original"], key, start_idx,
                original_np[:real_depth], mask_np[:real_depth]
            )
            add_patch(
                accumulators["dropout"], key, start_idx,
                dropout_np[:real_depth], mask_np[:real_depth]
            )
            add_patch(
                accumulators["ensemble"], key, start_idx,
                ensemble_np[:real_depth], mask_np[:real_depth]
            )

            if batch_index % 100 == 0 or batch_index == len(loader):
                print(f"  Processed {batch_index}/{len(loader)} patches")

    # -----------------------------------------------------------------
    # Reconstruct each unique slice and calculate per-slice/per-nodule metrics
    # -----------------------------------------------------------------
    nodule_rows = []
    slice_rows = []
    comparison_rows = []

    model_names = ["original", "dropout", "ensemble"]
    overall_counts = {
        name: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for name in model_names
    }

    all_keys = sorted(accumulators["original"].keys())

    for key in all_keys:
        ref_by_slice = accumulators["original"][key]["reference"]
        sorted_slices = sorted(ref_by_slice.keys())
        reference_volume = np.stack([ref_by_slice[s] for s in sorted_slices])

        for model_name in model_names:
            item = accumulators[model_name][key]
            probability_volume = np.stack([
                item["prob_sum"][s] / item["prob_count"][s]
                for s in sorted_slices
            ])
            prediction_volume = probability_volume >= THRESHOLD

            nodule_metrics = metric_counts(reference_volume, prediction_volume)
            patient_id, nodule_id = key.split("__", 1)

            nodule_rows.append({
                "patient_id": patient_id,
                "nodule_id": nodule_id,
                "model": model_name,
                "num_slices": len(sorted_slices),
                "reference_positive_voxels": nodule_metrics["reference_positive_voxels"],
                "predicted_positive_voxels": nodule_metrics["predicted_positive_voxels"],
                "dice": nodule_metrics["dice"],
                "iou": nodule_metrics["iou"],
                "precision": nodule_metrics["precision"],
                "recall": nodule_metrics["recall"],
                "specificity": nodule_metrics["specificity"],
                "accuracy": nodule_metrics["accuracy"],
            })

            for count_name in ("tp", "fp", "fn", "tn"):
                overall_counts[model_name][count_name] += nodule_metrics[count_name]

            for local_index, slice_index in enumerate(sorted_slices):
                slice_metrics = metric_counts(
                    reference_volume[local_index],
                    prediction_volume[local_index],
                )
                slice_rows.append({
                    "patient_id": patient_id,
                    "nodule_id": nodule_id,
                    "slice_position": slice_index,
                    "model": model_name,
                    "dice": slice_metrics["dice"],
                    "iou": slice_metrics["iou"],
                    "precision": slice_metrics["precision"],
                    "recall": slice_metrics["recall"],
                    "specificity": slice_metrics["specificity"],
                    "accuracy": slice_metrics["accuracy"],
                    "reference_positive_voxels": slice_metrics["reference_positive_voxels"],
                    "predicted_positive_voxels": slice_metrics["predicted_positive_voxels"],
                })

    nodule_df = pd.DataFrame(nodule_rows)
    slice_df = pd.DataFrame(slice_rows)

    for model_name in model_names:
        counts = overall_counts[model_name]
        tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]

        # Micro/voxel-level metrics over all reconstructed validation nodules.
        micro = {
            "dice": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0,
            "iou": tp / (tp + fp + fn) if (tp + fp + fn) else 1.0,
            "precision": tp / (tp + fp) if (tp + fp) else 1.0,
            "recall": tp / (tp + fn) if (tp + fn) else 1.0,
            "specificity": tn / (tn + fp) if (tn + fp) else 1.0,
            "accuracy": (tp + tn) / (tp + tn + fp + fn)
                        if (tp + tn + fp + fn) else 1.0,
        }

        model_nodules = nodule_df[nodule_df["model"] == model_name]
        annotated = model_nodules[model_nodules["reference_positive_voxels"] > 0]
        empty_reference_count = int(
            (model_nodules["reference_positive_voxels"] == 0).sum()
        )

        comparison_rows.append({
            "model": model_name,
            "num_nodules": len(model_nodules),
            "annotated_nodules": len(annotated),
            "empty_reference_nodules": empty_reference_count,
            "micro_dice": micro["dice"],
            "micro_iou": micro["iou"],
            "micro_precision": micro["precision"],
            "micro_recall": micro["recall"],
            "micro_specificity": micro["specificity"],
            "micro_accuracy": micro["accuracy"],
            "annotated_nodule_mean_dice": annotated["dice"].mean() if len(annotated) else np.nan,
            "annotated_nodule_median_dice": annotated["dice"].median() if len(annotated) else np.nan,
            "annotated_nodule_mean_iou": annotated["iou"].mean() if len(annotated) else np.nan,
            "annotated_nodule_mean_precision": annotated["precision"].mean() if len(annotated) else np.nan,
            "annotated_nodule_mean_recall": annotated["recall"].mean() if len(annotated) else np.nan,
        })

    comparison_df = pd.DataFrame(comparison_rows)

    # Save outputs
    nodule_path = OUTPUT_DIR / "ensemble_per_nodule_metrics.csv"
    slice_path = OUTPUT_DIR / "ensemble_per_slice_metrics.csv"
    comparison_path = OUTPUT_DIR / "ensemble_model_comparison.csv"
    summary_path = OUTPUT_DIR / "ensemble_evaluation_summary.txt"

    nodule_df.to_csv(nodule_path, index=False)
    slice_df.to_csv(slice_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("3D U-Net Ensemble Validation Evaluation\n")
        file.write("=" * 55 + "\n")
        file.write(f"Threshold: {THRESHOLD}\n")
        file.write(f"Validation nodules: {len(all_keys)}\n\n")
        file.write(comparison_df.to_string(index=False))
        file.write("\n\nNote: annotated-nodule macro metrics exclude nodules whose reference mask is empty.\n")
        file.write("Micro metrics aggregate voxel counts across all reconstructed nodules.\n")

    print("\n" + "=" * 78)
    print("VALIDATION COMPARISON")
    print("=" * 78)
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(comparison_df.round(5).to_string(index=False))

    print("\nSaved results:")
    print(" ", nodule_path)
    print(" ", slice_path)
    print(" ", comparison_path)
    print(" ", summary_path)
    print("\nEvaluation complete. No checkpoint was changed.")


if __name__ == "__main__":
    evaluate()
