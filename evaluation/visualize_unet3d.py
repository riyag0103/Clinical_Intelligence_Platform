import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast


# ============================================================
# 1. PROJECT CONFIGURATION
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
    / "visualizations"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.5
NUM_CASES = 10
NUM_WORKERS = 0

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

use_amp = device.type == "cuda"


# ============================================================
# 2. STARTUP INFORMATION
# ============================================================

print("=" * 70)
print("3D U-NET SEGMENTATION VISUALIZATION AND DIAGNOSTICS")
print("=" * 70)

print(f"Project root: {PROJECT_ROOT}")
print(f"Device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print(f"Prediction threshold: {THRESHOLD}")
print(f"Visualizations requested: {NUM_CASES}")


# ============================================================
# 3. CHECK REQUIRED FILES
# ============================================================

if not TEST_CSV.exists():
    raise FileNotFoundError(
        f"Test CSV does not exist:\n{TEST_CSV}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model checkpoint does not exist:\n{MODEL_PATH}"
    )


# ============================================================
# 4. IMPORT MODEL AND DATASET
# ============================================================

from models.unet3d import UNet3D
from data_loader.dataset_3d_patches import CT3DPatchDataset


# ============================================================
# 5. LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained 3D U-Net...")

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

    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")

    if "val_dice" in checkpoint:
        print(
            f"Checkpoint validation Dice: "
            f"{checkpoint['val_dice']:.4f}"
        )

else:
    model.load_state_dict(checkpoint)

model.eval()

print("Model loaded successfully.")


# ============================================================
# 6. LOAD TEST DATASET
# ============================================================

print("\nLoading test dataset...")

dataset = CT3DPatchDataset(
    csv_file=str(TEST_CSV),
    image_root=str(IMAGE_ROOT),
    mask_root=str(MASK_ROOT),
    patch_depth=8,
    stride=4
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=NUM_WORKERS
)

print(f"Total test patches: {len(dataset)}")


# ============================================================
# 7. HELPER FUNCTIONS
# ============================================================

def to_numpy(tensor):
    """Convert a PyTorch tensor into a NumPy array."""
    return tensor.detach().float().cpu().numpy()


def normalize_for_display(image):
    """
    Normalize a CT slice for visualization only.
    The model input remains unchanged.
    """
    image = image.astype(np.float32)

    low = np.percentile(image, 1)
    high = np.percentile(image, 99)

    if high <= low:
        return np.zeros_like(image)

    image = np.clip(image, low, high)

    return (image - low) / (high - low)


def calculate_dice(reference, prediction):
    """
    Calculate binary Dice for a single slice.

    Empty reference and prediction masks produce Dice = 1
    by convention. Such cases must not be interpreted as
    evidence of successful tumor segmentation.
    """
    reference = reference.astype(bool)
    prediction = prediction.astype(bool)

    intersection = np.logical_and(
        reference,
        prediction
    ).sum()

    reference_count = reference.sum()
    prediction_count = prediction.sum()

    denominator = reference_count + prediction_count

    if denominator == 0:
        return 1.0, True

    dice = (
        2.0 * intersection
    ) / denominator

    return float(dice), False


# ============================================================
# 8. VISUALIZE AND DIAGNOSE TEST PATCHES
# ============================================================

saved_cases = 0
examined_patches = 0

positive_reference_slices = 0
positive_prediction_slices = 0
empty_reference_and_prediction_slices = 0

with torch.inference_mode():

    for patch_index, batch in enumerate(loader):

        if saved_cases >= NUM_CASES:
            break

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

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        with autocast(
            device_type="cuda",
            enabled=use_amp
        ):
            logits = model(images)

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= THRESHOLD
        ).float()

        # ----------------------------------------------------
        # CONVERT TO NUMPY
        # ----------------------------------------------------

        image_volume = to_numpy(
            images[0, 0]
        )

        reference_volume = to_numpy(
            masks[0, 0]
        )

        prediction_volume = to_numpy(
            predictions[0, 0]
        )

        probability_volume = to_numpy(
            probabilities[0, 0]
        )

        valid = to_numpy(
            valid_slices[0]
        ).astype(bool)

        valid_indices = np.flatnonzero(valid)

        if len(valid_indices) == 0:
            print(
                f"Skipping patch {patch_index + 1}: "
                "no valid slices."
            )
            continue

        examined_patches += 1

        # ----------------------------------------------------
        # MASK DIAGNOSTICS
        # ----------------------------------------------------

        reference_counts = {
            int(i): int(
                np.sum(reference_volume[i] >= 0.5)
            )
            for i in valid_indices
        }

        prediction_counts = {
            int(i): int(
                np.sum(prediction_volume[i] >= 0.5)
            )
            for i in valid_indices
        }

        print("\n" + "-" * 70)
        print(f"PATCH {patch_index + 1} DIAGNOSTICS")
        print("-" * 70)

        print("Valid slice indices:", valid_indices.tolist())

        print(
            "Reference positive pixels per slice:",
            reference_counts
        )

        print(
            "Predicted positive pixels per slice:",
            prediction_counts
        )

        print(
            "Maximum probability in patch:",
            f"{float(np.max(probability_volume[valid_indices])):.6f}"
        )

        print(
            "Minimum probability in patch:",
            f"{float(np.min(probability_volume[valid_indices])):.6f}"
        )

        print(
            "Mean probability in valid slices:",
            f"{float(np.mean(probability_volume[valid_indices])):.6f}"
        )

        # ----------------------------------------------------
        # SELECT A SLICE WITH A REFERENCE NODULE
        # ----------------------------------------------------

        positive_indices = [
            int(i)
            for i in valid_indices
            if reference_counts[int(i)] > 0
        ]

        if positive_indices:

            slice_index = max(
                positive_indices,
                key=lambda i: reference_counts[i]
            )

        else:

            # For diagnostic purposes, choose the valid slice
            # with the largest predicted foreground region.
            slice_index = max(
                valid_indices.tolist(),
                key=lambda i: prediction_counts[int(i)]
            )

        image_slice = normalize_for_display(
            image_volume[slice_index]
        )

        reference_mask = (
            reference_volume[slice_index] >= 0.5
        )

        prediction_mask = (
            prediction_volume[slice_index] >= 0.5
        )

        probability_map = probability_volume[slice_index]

        reference_pixels = int(reference_mask.sum())
        predicted_pixels = int(prediction_mask.sum())

        dice, both_empty = calculate_dice(
            reference_mask,
            prediction_mask
        )

        if reference_pixels > 0:
            positive_reference_slices += 1

        if predicted_pixels > 0:
            positive_prediction_slices += 1

        if both_empty:
            empty_reference_and_prediction_slices += 1

        print(f"Selected slice: {slice_index}")
        print(f"Reference positive pixels: {reference_pixels}")
        print(f"Predicted positive pixels: {predicted_pixels}")
        print(f"Slice Dice: {dice:.4f}")

        if both_empty:
            print(
                "WARNING: Both masks are empty. "
                "Dice = 1.0 here does not demonstrate "
                "successful nodule segmentation."
            )

        if reference_pixels > 0 and predicted_pixels == 0:
            print(
                "WARNING: The reference contains a nodule, "
                "but the model predicts no positive pixels "
                "at the current threshold."
            )

        if reference_pixels == 0 and predicted_pixels > 0:
            print(
                "NOTE: The model predicts foreground, "
                "but this slice has an empty reference mask."
            )

        # ----------------------------------------------------
        # CREATE VISUALIZATION
        # ----------------------------------------------------

        fig, axes = plt.subplots(
            1,
            4,
            figsize=(18, 5)
        )

        # Panel 1: Original image
        axes[0].imshow(
            image_slice,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[0].set_title(
            "Original CT Slice"
        )

        # Panel 2: Reference mask
        axes[1].imshow(
            image_slice,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        if reference_pixels > 0:
            reference_overlay = np.ma.masked_where(
                ~reference_mask,
                reference_mask
            )

            axes[1].imshow(
                reference_overlay,
                cmap="Greens",
                alpha=0.75,
                vmin=0,
                vmax=1
            )
        else:
            axes[1].text(
                0.5,
                0.5,
                "EMPTY REFERENCE MASK",
                transform=axes[1].transAxes,
                ha="center",
                va="center",
                color="red",
                fontsize=10,
                bbox=dict(
                    facecolor="white",
                    alpha=0.8
                )
            )

        axes[1].set_title(
            f"Reference Mask\nPixels: {reference_pixels}"
        )

        # Panel 3: Predicted mask
        axes[2].imshow(
            image_slice,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        if predicted_pixels > 0:
            prediction_overlay = np.ma.masked_where(
                ~prediction_mask,
                prediction_mask
            )

            axes[2].imshow(
                prediction_overlay,
                cmap="Reds",
                alpha=0.75,
                vmin=0,
                vmax=1
            )
        else:
            axes[2].text(
                0.5,
                0.5,
                "NO POSITIVE PREDICTION",
                transform=axes[2].transAxes,
                ha="center",
                va="center",
                color="red",
                fontsize=10,
                bbox=dict(
                    facecolor="white",
                    alpha=0.8
                )
            )

        axes[2].set_title(
            f"Predicted Mask\nPixels: {predicted_pixels}"
        )

        # Panel 4: Probability map
        probability_plot = axes[3].imshow(
            probability_map,
            cmap="viridis",
            vmin=0,
            vmax=1
        )

        axes[3].set_title(
            "Prediction Probability"
        )

        fig.colorbar(
            probability_plot,
            ax=axes[3],
            fraction=0.046,
            pad=0.04
        )

        for ax in axes:
            ax.axis("off")

        fig.suptitle(
            f"Test Patch {patch_index + 1} | "
            f"Slice {slice_index} | "
            f"Dice: {dice:.4f}"
            + (
                " | Both masks empty"
                if both_empty
                else ""
            ),
            fontsize=13
        )

        fig.tight_layout()

        output_path = (
            OUTPUT_DIR
            / f"test_patch_{patch_index + 1:03d}.png"
        )

        fig.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close(fig)

        saved_cases += 1

        print(f"Saved: {output_path}")

        # ----------------------------------------------------
        # PER-PATCH SUMMARY
        # ----------------------------------------------------

        print(
            f"Visualizations generated: "
            f"{saved_cases}/{NUM_CASES}"
        )


# ============================================================
# 9. FINAL DIAGNOSTIC SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("VISUALIZATION AND DIAGNOSTICS COMPLETE")
print("=" * 70)

print(f"Patches examined: {examined_patches}")
print(f"Images saved: {saved_cases}")

print(
    "Selected slices with positive reference masks:",
    positive_reference_slices
)

print(
    "Selected slices with positive predictions:",
    positive_prediction_slices
)

print(
    "Selected slices where both masks were empty:",
    empty_reference_and_prediction_slices
)

print(f"\nOutput directory:\n{OUTPUT_DIR}")

if saved_cases == 0:
    print(
        "\nNo visualizations were created. "
        "Check the dataset and checkpoint paths."
    )
else:
    print(
        "\nOpen the PNG images in the output directory. "
        "The terminal diagnostics will help explain "
        "empty masks and Dice scores."
    )