from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------
# PROJECT CONFIGURATION
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(r"D:\Clinical_Project")

TEST_CSV = PROJECT_ROOT / "datasets" / "metadata" / "3d_splits" / "test_3d.csv"
IMAGE_ROOT = PROJECT_ROOT / "datasets" / "processed" / "ct"
MASK_ROOT = PROJECT_ROOT / "datasets" / "processed" / "lidc_consensus_masks"

CHECKPOINT_PATH = (
    PROJECT_ROOT / "saved_models" / "unet3d" / "best_unet3d_dropout.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "unet3d" / "reconstructed_test_evaluation"
)
VIS_DIR = OUTPUT_DIR / "visualizations"

PATCH_DEPTH = 8
PATCH_STRIDE = 4
THRESHOLD = 0.15
BATCH_SIZE = 1
NUM_WORKERS = 0
MAX_VISUALIZATIONS = 30
EPS = 1e-8

# If your training script used a different patch stride, change this to match.
# The dataset class creates the patch locations from PATCH_DEPTH and PATCH_STRIDE.


def add_project_to_path():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def load_model(device):
    add_project_to_path()
    from models.unet3d import UNet3D

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    model = UNet3D(in_channels=1, out_channels=1).to(device)

    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print("Loaded checkpoint:", CHECKPOINT_PATH)
    if isinstance(checkpoint, dict):
        print("Checkpoint epoch:", checkpoint.get("epoch", "not stored"))
        print("Checkpoint validation Dice:", checkpoint.get("val_dice", "not stored"))

    return model


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def get_slice_number(path):
    match = re.search(r"slice-(\d+)", Path(path).stem)
    return int(match.group(1)) if match else None


def make_nodule_record(patient_id, nodule_id):
    return {
        "patient_id": str(patient_id),
        "nodule_id": str(nodule_id),
        "probability_sum": {},
        "probability_count": {},
        "ground_truth": {},
        "images": {},
        "slice_names": {},
    }


def add_patch_to_record(record, sample, probabilities, masks, images):
    """
    Add one patch's per-slice outputs to its nodule record.

    Dataset start_idx is the index into the sorted image_paths list.
    We use that same index to reconstruct each nodule's ordered slice stack.
    """
    start_idx = int(sample["start_idx"])
    real_depth = int(sample["real_depth"])
    image_paths = sample["image_paths"]

    for local_idx in range(real_depth):
        stack_idx = start_idx + local_idx

        prob_slice = probabilities[local_idx].astype(np.float32)
        gt_slice = (masks[local_idx] > 0.5).astype(np.uint8)
        image_slice = images[local_idx].astype(np.float32)

        if stack_idx not in record["probability_sum"]:
            record["probability_sum"][stack_idx] = np.zeros_like(
                prob_slice, dtype=np.float64
            )
            record["probability_count"][stack_idx] = 0
            record["ground_truth"][stack_idx] = gt_slice
            record["images"][stack_idx] = image_slice
            record["slice_names"][stack_idx] = image_paths[local_idx].name
        else:
            # The same original slice appears in overlapping patches.
            # Its ground-truth mask should be identical in every patch.
            existing_gt = record["ground_truth"][stack_idx]
            if existing_gt.shape != gt_slice.shape or not np.array_equal(existing_gt, gt_slice):
                raise RuntimeError(
                    "Ground-truth mask mismatch across overlapping patches for "
                    f"{record['patient_id']} / {record['nodule_id']} / "
                    f"{record['slice_names'][stack_idx]}"
                )

        record["probability_sum"][stack_idx] += prob_slice
        record["probability_count"][stack_idx] += 1


@torch.inference_mode()
def infer_and_reconstruct(model, dataset, device):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    records = {}
    total_batches = len(loader)

    for batch_idx, batch in enumerate(loader):
        images_t = batch["image"].to(device, non_blocking=True)
        logits = model(images_t)
        probs_t = torch.sigmoid(logits)

        probs = probs_t[0, 0].detach().cpu().numpy()
        imgs = batch["image"][0, 0].cpu().numpy()
        gts = batch["mask"][0, 0].cpu().numpy()

        # With batch size 1, use the original dataset sample for exact paths
        # and integer metadata rather than relying on default collation types.
        sample = dataset.samples[batch_idx]
        patient_id = sample["patient_id"]
        nodule_id = sample["nodule_id"]
        key = (str(patient_id), str(nodule_id))

        if key not in records:
            records[key] = make_nodule_record(*key)

        add_patch_to_record(
            records[key],
            sample,
            probs,
            gts,
            imgs,
        )

        if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == total_batches:
            print(f"Processed patches: {batch_idx + 1}/{total_batches}")

    return records


def calculate_metrics_for_nodule(record):
    slice_indices = sorted(record["probability_sum"].keys())

    pred_slices = []
    gt_slices = []
    image_slices = []
    probability_slices = []

    for idx in slice_indices:
        avg_prob = (
            record["probability_sum"][idx]
            / max(record["probability_count"][idx], 1)
        ).astype(np.float32)

        pred = (avg_prob >= THRESHOLD).astype(np.uint8)
        gt = record["ground_truth"][idx].astype(np.uint8)

        probability_slices.append(avg_prob)
        pred_slices.append(pred)
        gt_slices.append(gt)
        image_slices.append(record["images"][idx])

    pred_volume = np.stack(pred_slices, axis=0).astype(bool)
    gt_volume = np.stack(gt_slices, axis=0).astype(bool)
    probability_volume = np.stack(probability_slices, axis=0)
    image_volume = np.stack(image_slices, axis=0)

    tp = int(np.logical_and(pred_volume, gt_volume).sum())
    fp = int(np.logical_and(pred_volume, ~gt_volume).sum())
    fn = int(np.logical_and(~pred_volume, gt_volume).sum())
    tn = int(np.logical_and(~pred_volume, ~gt_volume).sum())

    gt_pixels = int(gt_volume.sum())
    pred_pixels = int(pred_volume.sum())

    dice = (
        safe_ratio(2 * tp, 2 * tp + fp + fn)
        if gt_pixels > 0
        else np.nan
    )
    iou = (
        safe_ratio(tp, tp + fp + fn)
        if gt_pixels > 0
        else np.nan
    )

    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn) if gt_pixels > 0 else np.nan
    specificity = safe_ratio(tn, tn + fp)

    return {
        "patient_id": record["patient_id"],
        "nodule_id": record["nodule_id"],
        "reconstructed_slices": len(slice_indices),
        "gt_pixels": gt_pixels,
        "predicted_pixels": pred_pixels,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "empty_ground_truth": gt_pixels == 0,
        "empty_prediction": pred_pixels == 0,
        "probability_volume": probability_volume,
        "prediction_volume": pred_volume.astype(np.uint8),
        "ground_truth_volume": gt_volume.astype(np.uint8),
        "image_volume": image_volume,
        "slice_indices": slice_indices,
    }


def save_visualization(result, output_path):
    gt = result["ground_truth_volume"]
    pred = result["prediction_volume"]
    probs = result["probability_volume"]
    imgs = result["image_volume"]

    # Select slice with largest GT foreground area; if GT is empty,
    # select the slice with the largest predicted area.
    gt_areas = gt.reshape(gt.shape[0], -1).sum(axis=1)
    if gt_areas.max() > 0:
        selected = int(np.argmax(gt_areas))
    else:
        pred_areas = pred.reshape(pred.shape[0], -1).sum(axis=1)
        selected = int(np.argmax(pred_areas))

    pred_masked = np.ma.masked_where(pred[selected] == 0, pred[selected])

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(imgs[selected], cmap="gray")
    axes[0].set_title("Original CT")

    axes[1].imshow(gt[selected], cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")

    axes[2].imshow(probs[selected], cmap="hot", vmin=0, vmax=1)
    axes[2].set_title("Averaged Probability")

    axes[3].imshow(imgs[selected], cmap="gray")
    axes[3].imshow(
        pred_masked,
        cmap="autumn",
        alpha=0.75,
        vmin=0,
        vmax=1,
    )
    axes[3].set_title("Prediction Overlay")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{result['patient_id']} | {result['nodule_id']} | "
        f"reconstructed slice {result['slice_indices'][selected]} | "
        f"Dice {result['dice'] if not np.isnan(result['dice']) else 'N/A'}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    for path in [TEST_CSV, IMAGE_ROOT, MASK_ROOT, CHECKPOINT_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    add_project_to_path()
    from data_loader.dataset_3d_patches import CT3DPatchDataset

    test_metadata = pd.read_csv(TEST_CSV)
    print("=" * 76)
    print("RECONSTRUCTED PER-NODULE 3D U-NET TEST EVALUATION")
    print("=" * 76)
    print("Metadata nodule rows:", len(test_metadata))
    print("Unique patients:", test_metadata["patient_id"].nunique())
    print("Unique nodules:", test_metadata[["patient_id", "nodule_id"]].drop_duplicates().shape[0])
    print("Threshold:", THRESHOLD)
    print("Patch depth:", PATCH_DEPTH)
    print("Patch stride:", PATCH_STRIDE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    dataset = CT3DPatchDataset(
        csv_file=str(TEST_CSV),
        image_root=str(IMAGE_ROOT),
        mask_root=str(MASK_ROOT),
        patch_depth=PATCH_DEPTH,
        stride=PATCH_STRIDE,
    )
    print("Patches to evaluate:", len(dataset))

    model = load_model(device)
    records = infer_and_reconstruct(model, dataset, device)

    metric_rows = []
    detailed_results = {}

    total_tp = total_fp = total_fn = total_tn = 0
    visual_count = 0

    for key, record in records.items():
        result = calculate_metrics_for_nodule(record)

        row = {
            k: v for k, v in result.items()
            if k not in {
                "probability_volume",
                "prediction_volume",
                "ground_truth_volume",
                "image_volume",
                "slice_indices",
            }
        }
        metric_rows.append(row)

        total_tp += result["tp"]
        total_fp += result["fp"]
        total_fn += result["fn"]
        total_tn += result["tn"]

        # Save representative visualizations for positive-GT nodules first.
        if result["gt_pixels"] > 0 and visual_count < MAX_VISUALIZATIONS:
            filename = (
                f"{result['patient_id']}_{result['nodule_id']}"
                .replace("/", "_")
                .replace("\\", "_")
                + ".png"
            )
            save_visualization(result, VIS_DIR / filename)
            visual_count += 1

        detailed_results[f"{key[0]}::{key[1]}"] = {
            "slice_indices": result["slice_indices"],
            "probability_volume": result["probability_volume"].tolist(),
            "prediction_volume": result["prediction_volume"].tolist(),
            "ground_truth_volume": result["ground_truth_volume"].tolist(),
        }

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(OUTPUT_DIR / "per_nodule_reconstructed_metrics.csv", index=False)

    positive_df = metrics_df[metrics_df["gt_pixels"] > 0].copy()
    empty_gt_df = metrics_df[metrics_df["gt_pixels"] == 0].copy()

    # Micro metrics over reconstructed, unique nodule-slice voxels.
    micro_dice = safe_ratio(2 * total_tp, 2 * total_tp + total_fp + total_fn)
    micro_iou = safe_ratio(total_tp, total_tp + total_fp + total_fn)
    micro_precision = safe_ratio(total_tp, total_tp + total_fp)
    micro_recall = safe_ratio(total_tp, total_tp + total_fn)
    micro_specificity = safe_ratio(total_tn, total_tn + total_fp)
    micro_accuracy = safe_ratio(
        total_tp + total_tn,
        total_tp + total_fp + total_fn + total_tn,
    )

    summary = {
        "threshold": THRESHOLD,
        "patch_depth": PATCH_DEPTH,
        "patch_stride": PATCH_STRIDE,
        "patches_evaluated": len(dataset),
        "nodules_reconstructed": len(metrics_df),
        "positive_ground_truth_nodules": len(positive_df),
        "empty_ground_truth_nodules": len(empty_gt_df),
        "positive_nodules_with_zero_dice": int((positive_df["dice"] == 0).sum()),
        "positive_nodules_dice_ge_0_70": int((positive_df["dice"] >= 0.70).sum()),
        "positive_nodules_dice_ge_0_50": int((positive_df["dice"] >= 0.50).sum()),
        "mean_per_nodule_dice_positive_gt": float(positive_df["dice"].mean()) if len(positive_df) else None,
        "median_per_nodule_dice_positive_gt": float(positive_df["dice"].median()) if len(positive_df) else None,
        "std_per_nodule_dice_positive_gt": float(positive_df["dice"].std()) if len(positive_df) > 1 else None,
        "mean_per_nodule_iou_positive_gt": float(positive_df["iou"].mean()) if len(positive_df) else None,
        "median_per_nodule_iou_positive_gt": float(positive_df["iou"].median()) if len(positive_df) else None,
        "empty_gt_nodules_with_any_prediction": int((empty_gt_df["predicted_pixels"] > 0).sum()),
        "empty_gt_nodules_with_no_prediction": int((empty_gt_df["predicted_pixels"] == 0).sum()),
        "micro_dice_unique_reconstructed_voxels": micro_dice,
        "micro_iou_unique_reconstructed_voxels": micro_iou,
        "micro_precision_unique_reconstructed_voxels": micro_precision,
        "micro_recall_unique_reconstructed_voxels": micro_recall,
        "micro_specificity_unique_reconstructed_voxels": micro_specificity,
        "micro_accuracy_unique_reconstructed_voxels": micro_accuracy,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_tn": total_tn,
        "visualizations_saved": visual_count,
    }

    with open(OUTPUT_DIR / "reconstructed_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    pd.DataFrame([summary]).to_csv(
        OUTPUT_DIR / "reconstructed_evaluation_summary.csv",
        index=False,
    )

    print("\n" + "=" * 76)
    print("RECONSTRUCTED TEST RESULTS")
    print("=" * 76)
    for key, value in summary.items():
        print(f"{key:48s}: {value}")

    print("\nFiles saved:")
    print(OUTPUT_DIR / "per_nodule_reconstructed_metrics.csv")
    print(OUTPUT_DIR / "reconstructed_evaluation_summary.csv")
    print(OUTPUT_DIR / "reconstructed_evaluation_summary.json")
    print(VIS_DIR)

    print(
        "\nNOTE: Predictions from overlapping patches are averaged per original "
        "slice before thresholding. Each reconstructed nodule slice is counted "
        "once within that nodule. If different nodule annotations overlap in "
        "the same patient slice, this script evaluates them as separate nodule "
        "records rather than merging them into one patient-level lung volume."
    )
    print("Evaluation complete. Do not tune the threshold on this held-out test set.")


if __name__ == "__main__":
    main()
