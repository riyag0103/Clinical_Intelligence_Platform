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


# ============================================================
# PROJECT CONFIGURATION
# ============================================================
PROJECT_ROOT = Path(r"D:\Clinical_Project")

VAL_CSV = (
    PROJECT_ROOT / "datasets" / "metadata" / "3d_splits" / "val_3d.csv"
)
IMAGE_ROOT = PROJECT_ROOT / "datasets" / "processed" / "ct"
MASK_ROOT = PROJECT_ROOT / "datasets" / "processed" / "lidc_consensus_masks"

CHECKPOINT_PATH = (
    PROJECT_ROOT / "saved_models" / "unet3d" / "best_unet3d_dropout.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "unet3d" / "validation_threshold_optimization"
)

PATCH_DEPTH = 8
PATCH_STRIDE = 4
BATCH_SIZE = 1
NUM_WORKERS = 0
EPS = 1e-8

THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


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
            CHECKPOINT_PATH, map_location=device, weights_only=False
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

    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")
    if isinstance(checkpoint, dict):
        print("Checkpoint epoch:", checkpoint.get("epoch", "not stored"))
        print("Checkpoint validation Dice:", checkpoint.get("val_dice", "not stored"))
    return model


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def make_record(patient_id, nodule_id):
    return {
        "patient_id": str(patient_id),
        "nodule_id": str(nodule_id),
        "probability_sum": {},
        "probability_count": {},
        "ground_truth": {},
    }


def add_patch(record, sample, probabilities, masks):
    start_idx = int(sample["start_idx"])
    real_depth = int(sample["real_depth"])

    for local_idx in range(real_depth):
        original_idx = start_idx + local_idx
        prob_slice = probabilities[local_idx].astype(np.float32)
        gt_slice = (masks[local_idx] > 0.5).astype(np.uint8)

        if original_idx not in record["probability_sum"]:
            record["probability_sum"][original_idx] = np.zeros_like(
                prob_slice, dtype=np.float64
            )
            record["probability_count"][original_idx] = 0
            record["ground_truth"][original_idx] = gt_slice
        else:
            existing_gt = record["ground_truth"][original_idx]
            if not np.array_equal(existing_gt, gt_slice):
                raise RuntimeError(
                    "Ground-truth mismatch across overlapping patches: "
                    f"{record['patient_id']} / {record['nodule_id']} / "
                    f"slice index {original_idx}"
                )

        record["probability_sum"][original_idx] += prob_slice
        record["probability_count"][original_idx] += 1


@torch.inference_mode()
def reconstruct_validation_predictions(model, dataset, device):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    records = {}
    total = len(loader)

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)
        probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()
        masks = batch["mask"][0, 0].cpu().numpy()

        # Batch size is one; use the dataset sample for exact metadata.
        sample = dataset.samples[batch_idx]
        key = (str(sample["patient_id"]), str(sample["nodule_id"]))

        if key not in records:
            records[key] = make_record(*key)

        add_patch(records[key], sample, probabilities, masks)

        if (batch_idx + 1) % 25 == 0 or batch_idx + 1 == total:
            print(f"Reconstructed patches: {batch_idx + 1}/{total}")

    # Average overlapping patch probabilities once per original slice.
    for record in records.values():
        record["probabilities"] = {}
        for idx, prob_sum in record["probability_sum"].items():
            count = max(record["probability_count"][idx], 1)
            record["probabilities"][idx] = (
                prob_sum / count
            ).astype(np.float32)

        record["ground_truth_sorted"] = {
            idx: record["ground_truth"][idx]
            for idx in sorted(record["ground_truth"])
        }
        record["probabilities_sorted"] = {
            idx: record["probabilities"][idx]
            for idx in sorted(record["probabilities"])
        }

    return records


def evaluate_threshold(records, threshold):
    total_tp = total_fp = total_fn = total_tn = 0
    per_nodule_dice = []
    per_nodule_iou = []
    positive_nodules = 0
    zero_dice_nodules = 0

    for record in records.values():
        indices = sorted(record["probabilities_sorted"])

        pred_volume = np.stack(
            [
                record["probabilities_sorted"][idx] >= threshold
                for idx in indices
            ],
            axis=0,
        ).astype(bool)

        gt_volume = np.stack(
            [
                record["ground_truth_sorted"][idx] > 0
                for idx in indices
            ],
            axis=0,
        ).astype(bool)

        tp = int(np.logical_and(pred_volume, gt_volume).sum())
        fp = int(np.logical_and(pred_volume, ~gt_volume).sum())
        fn = int(np.logical_and(~pred_volume, gt_volume).sum())
        tn = int(np.logical_and(~pred_volume, ~gt_volume).sum())

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        gt_count = int(gt_volume.sum())
        if gt_count > 0:
            positive_nodules += 1
            dice = safe_ratio(2 * tp, 2 * tp + fp + fn)
            iou = safe_ratio(tp, tp + fp + fn)
            per_nodule_dice.append(dice)
            per_nodule_iou.append(iou)
            if dice == 0:
                zero_dice_nodules += 1

    micro_dice = safe_ratio(
        2 * total_tp, 2 * total_tp + total_fp + total_fn
    )
    micro_iou = safe_ratio(
        total_tp, total_tp + total_fp + total_fn
    )
    precision = safe_ratio(total_tp, total_tp + total_fp)
    recall = safe_ratio(total_tp, total_tp + total_fn)
    specificity = safe_ratio(total_tn, total_tn + total_fp)
    accuracy = safe_ratio(
        total_tp + total_tn,
        total_tp + total_fp + total_fn + total_tn,
    )

    return {
        "threshold": threshold,
        "micro_dice": micro_dice,
        "micro_iou": micro_iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "mean_per_nodule_dice": float(np.mean(per_nodule_dice))
        if per_nodule_dice else 0.0,
        "median_per_nodule_dice": float(np.median(per_nodule_dice))
        if per_nodule_dice else 0.0,
        "mean_per_nodule_iou": float(np.mean(per_nodule_iou))
        if per_nodule_iou else 0.0,
        "positive_nodules": positive_nodules,
        "zero_dice_positive_nodules": zero_dice_nodules,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    required = [VAL_CSV, IMAGE_ROOT, MASK_ROOT, CHECKPOINT_PATH]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    add_project_to_path()
    from data_loader.dataset_3d_patches import CT3DPatchDataset

    print("=" * 72)
    print("3D U-NET VALIDATION THRESHOLD OPTIMIZATION")
    print("=" * 72)
    print("Validation CSV:", VAL_CSV)
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Thresholds:", THRESHOLDS)
    print("Patch depth / stride:", PATCH_DEPTH, "/", PATCH_STRIDE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    dataset = CT3DPatchDataset(
        csv_file=str(VAL_CSV),
        image_root=str(IMAGE_ROOT),
        mask_root=str(MASK_ROOT),
        patch_depth=PATCH_DEPTH,
        stride=PATCH_STRIDE,
    )
    print("Validation patches:", len(dataset))

    model = load_model(device)

    # Run inference only once. Reuse probability maps for every threshold.
    records = reconstruct_validation_predictions(model, dataset, device)
    print("Reconstructed nodules:", len(records))

    rows = [
        evaluate_threshold(records, threshold)
        for threshold in THRESHOLDS
    ]
    results = pd.DataFrame(rows)

    # Select by reconstructed micro Dice. Tie-break with mean per-nodule Dice.
    results = results.sort_values(
        ["micro_dice", "mean_per_nodule_dice"],
        ascending=[False, False],
    ).reset_index(drop=True)

    best = results.iloc[0].to_dict()
    best_threshold = float(best["threshold"])

    # Save complete table sorted by threshold for readability.
    display_results = results.sort_values("threshold").reset_index(drop=True)
    csv_path = OUTPUT_DIR / "validation_threshold_comparison.csv"
    display_results.to_csv(csv_path, index=False)

    summary = {
        "selected_threshold": best_threshold,
        "selection_rule": (
            "Highest reconstructed validation micro Dice; "
            "tie-break by mean per-nodule Dice."
        ),
        "validation_nodules_reconstructed": len(records),
        "validation_patches": len(dataset),
        "patch_depth": PATCH_DEPTH,
        "patch_stride": PATCH_STRIDE,
        "checkpoint": str(CHECKPOINT_PATH),
        "best_validation_metrics": best,
        "warning": (
            "Threshold selected using validation data only. "
            "Do not tune further on the held-out test set."
        ),
    }

    json_path = OUTPUT_DIR / "best_validation_threshold.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    # Plot reconstructed validation metrics by threshold.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        display_results["threshold"],
        display_results["micro_dice"],
        marker="o",
        label="Micro Dice",
    )
    ax.plot(
        display_results["threshold"],
        display_results["mean_per_nodule_dice"],
        marker="s",
        label="Mean per-nodule Dice",
    )
    ax.plot(
        display_results["threshold"],
        display_results["micro_iou"],
        marker="^",
        label="Micro IoU",
    )
    ax.axvline(
        best_threshold,
        linestyle="--",
        label=f"Selected threshold = {best_threshold:.2f}",
    )
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Score")
    ax.set_title("3D U-Net: Validation Threshold Optimization")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    plot_path = OUTPUT_DIR / "validation_threshold_comparison.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 72)
    print("VALIDATION THRESHOLD RESULTS")
    print("=" * 72)
    print(
        display_results[
            [
                "threshold",
                "micro_dice",
                "micro_iou",
                "precision",
                "recall",
                "mean_per_nodule_dice",
                "zero_dice_positive_nodules",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\nSELECTED THRESHOLD:", f"{best_threshold:.2f}")
    print("Best validation micro Dice:", f"{best['micro_dice']:.4f}")
    print("Best validation micro IoU:", f"{best['micro_iou']:.4f}")
    print("Precision:", f"{best['precision']:.4f}")
    print("Recall:", f"{best['recall']:.4f}")
    print("Mean per-nodule Dice:", f"{best['mean_per_nodule_dice']:.4f}")

    print("\nSaved:")
    print(csv_path)
    print(json_path)
    print(plot_path)
    print("\nNo model weights were changed.")
    print("Use the selected threshold for one final held-out test evaluation.")
    print("Do not select or modify the threshold using test-set results.")


if __name__ == "__main__":
    main()
