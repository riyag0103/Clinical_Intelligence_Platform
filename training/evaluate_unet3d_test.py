import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(r"D:\Clinical_Project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.unet3d import UNet3D
from data_loader.dataset_3d_patches import CT3DPatchDataset


# ---------------------------------------------------------------------
# Configuration: keep these consistent with the original training run.
# ---------------------------------------------------------------------
TEST_CSV = PROJECT_ROOT / "datasets" / "metadata" / "3d_splits" / "test_3d.csv"
IMAGE_ROOT = PROJECT_ROOT / "datasets" / "processed" / "ct"
MASK_ROOT = PROJECT_ROOT / "datasets" / "processed" / "lidc_consensus_masks"
CHECKPOINT_PATH = PROJECT_ROOT / "saved_models" / "unet3d" / "best_unet3d_dropout.pth"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "unet3d" / "final_test_evaluation"
PLOT_DIR = OUTPUT_DIR / "visualizations"

PATCH_DEPTH = 8
STRIDE = 4
BATCH_SIZE = 1
NUM_WORKERS = 0
THRESHOLD = 0.20
MAX_VISUALIZATIONS = 20
EPS = 1e-8


def load_model(device):
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

    if isinstance(checkpoint, dict):
        print("Checkpoint epoch:", checkpoint.get("epoch", "not stored"))
        print("Checkpoint validation Dice:", checkpoint.get("val_dice", "not stored"))

    return model


def safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else 0.0


@torch.inference_mode()
def evaluate(model, loader, device):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    patch_rows = []
    visualizations_saved = 0

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for batch_index, batch in enumerate(loader):
        images = batch["image"].to(device)
        targets = batch["mask"].to(device)
        valid_slices = batch["valid_slices"].to(device)

        logits = model(images)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= THRESHOLD).float()

        valid = valid_slices[:, None, :, None, None].bool()
        predictions = predictions * valid
        targets = targets * valid

        pred_bool = predictions.bool()
        target_bool = targets.bool()

        tp = int((pred_bool & target_bool).sum().item())
        fp = int((pred_bool & ~target_bool & valid).sum().item())
        fn = int((~pred_bool & target_bool & valid).sum().item())
        tn = int((~pred_bool & ~target_bool & valid).sum().item())

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        patch_dice = safe_divide(2 * tp, 2 * tp + fp + fn)
        patch_iou = safe_divide(tp, tp + fp + fn)

        patient_id = batch["patient_id"][0]
        nodule_id = batch["nodule_id"][0]
        start_idx = int(batch["start_idx"][0])
        real_depth = int(batch["real_depth"][0])

        patch_rows.append({
            "patient_id": patient_id,
            "nodule_id": nodule_id,
            "start_idx": start_idx,
            "real_depth": real_depth,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "dice": patch_dice,
            "iou": patch_iou,
        })

        # Save a representative slice: the valid slice with the largest
        # ground-truth foreground area. If GT is empty, use the middle slice.
        if visualizations_saved < MAX_VISUALIZATIONS:
            gt_np = targets[0, 0, :real_depth].cpu().numpy()
            pred_np = predictions[0, 0, :real_depth].cpu().numpy()
            prob_np = probabilities[0, 0, :real_depth].cpu().numpy()
            image_np = images[0, 0, :real_depth].cpu().numpy()

            areas = gt_np.reshape(real_depth, -1).sum(axis=1)
            chosen_slice = int(np.argmax(areas)) if areas.max() > 0 else real_depth // 2

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(image_np[chosen_slice], cmap="gray")
            axes[0].set_title("Original CT")
            axes[1].imshow(gt_np[chosen_slice], cmap="gray", vmin=0, vmax=1)
            axes[1].set_title("Ground Truth")
            axes[2].imshow(prob_np[chosen_slice], cmap="hot", vmin=0, vmax=1)
            axes[2].set_title("Probability Map")
            pred_overlay = np.ma.masked_where(
                pred_np[chosen_slice] == 0,
                pred_np[chosen_slice],
            )
            axes[3].imshow(image_np[chosen_slice], cmap="gray")
            axes[3].imshow(
                pred_overlay,
                cmap="autumn",
                alpha=0.7,
                vmin=0,
                vmax=1,
            )
            axes[3].set_title("Prediction Overlay")

            for ax in axes:
                ax.axis("off")

            fig.suptitle(
                f"{patient_id} | {nodule_id} | patch start {start_idx} | "
                f"slice {chosen_slice} | threshold {THRESHOLD}"
            )
            fig.tight_layout()
            fig.savefig(
                PLOT_DIR / f"patch_{batch_index + 1:03d}.png",
                dpi=140,
                bbox_inches="tight",
            )
            plt.close(fig)
            visualizations_saved += 1

        if (batch_index + 1) % 25 == 0:
            print(f"Evaluated {batch_index + 1}/{len(loader)} patches")

    dice = safe_divide(2 * total_tp, 2 * total_tp + total_fp + total_fn)
    iou = safe_divide(total_tp, total_tp + total_fp + total_fn)
    precision = safe_divide(total_tp, total_tp + total_fp)
    recall = safe_divide(total_tp, total_tp + total_fn)
    specificity = safe_divide(total_tn, total_tn + total_fp)
    accuracy = safe_divide(total_tp + total_tn, total_tp + total_fp + total_fn + total_tn)

    summary = {
        "threshold": THRESHOLD,
        "patches_evaluated": len(patch_rows),
        "true_positive": total_tp,
        "false_positive": total_fp,
        "false_negative": total_fn,
        "true_negative": total_tn,
        "micro_dice": dice,
        "micro_iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "visualizations_saved": visualizations_saved,
    }

    return summary, patch_rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for required in [TEST_CSV, IMAGE_ROOT, MASK_ROOT, CHECKPOINT_PATH]:
        if not required.exists():
            raise FileNotFoundError(f"Required path not found: {required}")

    test_metadata = pd.read_csv(TEST_CSV)
    print("=" * 72)
    print("HELD-OUT TEST EVALUATION — 3D U-NET")
    print("=" * 72)
    print("Test metadata rows:", len(test_metadata))
    print("Unique test patients:", test_metadata["patient_id"].nunique())
    print("Nodule records:", len(test_metadata))
    print("Nodule slice-count distribution:")
    print(test_metadata["num_slices"].describe().to_string())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    dataset = CT3DPatchDataset(
        csv_file=str(TEST_CSV),
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
        pin_memory=(device.type == "cuda"),
    )

    print("Test patches created:", len(dataset))
    model = load_model(device)

    summary, patch_rows = evaluate(model, loader, device)

    pd.DataFrame(patch_rows).to_csv(
        OUTPUT_DIR / "per_patch_metrics.csv",
        index=False,
    )
    pd.DataFrame([summary]).to_csv(
        OUTPUT_DIR / "test_metrics_summary.csv",
        index=False,
    )

    print("\n" + "=" * 72)
    print("HELD-OUT TEST RESULTS")
    print("=" * 72)
    for key, value in summary.items():
        print(f"{key:24s}: {value}")

    print("\nSaved:")
    print(OUTPUT_DIR / "test_metrics_summary.csv")
    print(OUTPUT_DIR / "per_patch_metrics.csv")
    print(PLOT_DIR)
    print("\nEvaluation complete. Do not tune the threshold on this test set.")


if __name__ == "__main__":
    main()
    