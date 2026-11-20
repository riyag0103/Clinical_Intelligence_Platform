import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(r"D:\Clinical_Project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.unet3d import UNet3D
from data_loader.dataset_3d_patches import CT3DPatchDataset


# ---------------------------- PATHS ---------------------------------
TRAIN_CSV = PROJECT_ROOT / "datasets" / "metadata" / "3d_splits" / "train_3d.csv"
VAL_CSV = PROJECT_ROOT / "datasets" / "metadata" / "3d_splits" / "val_3d.csv"
IMAGE_ROOT = PROJECT_ROOT / "datasets" / "processed" / "ct"
MASK_ROOT = PROJECT_ROOT / "datasets" / "processed" / "lidc_consensus_masks"

CHECKPOINT_DIR = PROJECT_ROOT / "saved_models" / "unet3d"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "unet3d" / "dropout" / "stable_resume"

SOURCE_CHECKPOINT = CHECKPOINT_DIR / "best_unet3d_dropout.pth"
STABLE_BEST_CHECKPOINT = CHECKPOINT_DIR / "best_unet3d_dropout_stable.pth"
LATEST_CHECKPOINT = CHECKPOINT_DIR / "last_unet3d_dropout_stable.pth"
HISTORY_PATH = OUTPUT_DIR / "stable_resume_history.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------- CONFIGURATION -----------------------------
# Short continuation only. Do not launch another 100-epoch run under
# the one-day submission deadline.
ADDITIONAL_EPOCHS = 5
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5
PATCH_DEPTH = 8
STRIDE = 4
NUM_WORKERS = 0
SEED = 42
THRESHOLD = 0.50

# Deliberately disable AMP for this diagnostic continuation.
USE_AMP = False

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint_file(path):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def check_finite_tensor(name, tensor):
    if not torch.isfinite(tensor).all():
        bad_count = int((~torch.isfinite(tensor)).sum().item())
        raise ValueError(f"{name} contains {bad_count} NaN/Inf values.")


# ------------------------------ DATA --------------------------------
for required in [TRAIN_CSV, VAL_CSV, IMAGE_ROOT, MASK_ROOT, SOURCE_CHECKPOINT]:
    if not required.exists():
        raise FileNotFoundError(f"Required path not found: {required}")

print("=" * 72)
print("STABLE DROPOUT 3D U-NET CONTINUATION")
print("=" * 72)
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("AMP: disabled (FP32 stability run)")
print(f"Source checkpoint: {SOURCE_CHECKPOINT}")
print(f"New best checkpoint: {STABLE_BEST_CHECKPOINT}")
print(f"Additional epochs: {ADDITIONAL_EPOCHS}")

train_dataset = CT3DPatchDataset(
    csv_file=str(TRAIN_CSV),
    image_root=str(IMAGE_ROOT),
    mask_root=str(MASK_ROOT),
    patch_depth=PATCH_DEPTH,
    stride=STRIDE,
)
val_dataset = CT3DPatchDataset(
    csv_file=str(VAL_CSV),
    image_root=str(IMAGE_ROOT),
    mask_root=str(MASK_ROOT),
    patch_depth=PATCH_DEPTH,
    stride=STRIDE,
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
)

print(f"Training patches: {len(train_dataset)}")
print(f"Validation patches: {len(val_dataset)}")


# -------------------------- MODEL/RESUME -----------------------------
model = UNet3D().to(device)
source_checkpoint = load_checkpoint_file(SOURCE_CHECKPOINT)
model.load_state_dict(extract_state_dict(source_checkpoint), strict=True)

source_epoch = int(source_checkpoint.get("epoch", 0)) if isinstance(source_checkpoint, dict) else 0
source_val_dice = float(source_checkpoint.get("val_dice", 0.0)) if isinstance(source_checkpoint, dict) else 0.0
source_val_iou = float(source_checkpoint.get("val_iou", 0.0)) if isinstance(source_checkpoint, dict) else 0.0

print(f"Loaded checkpoint epoch: {source_epoch}")
print(f"Checkpoint validation Dice: {source_val_dice:.6f}")
print(f"Checkpoint validation IoU: {source_val_iou:.6f}")

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

# Restore optimiser momentum/state when available, then set the deliberate
# lower learning rate for this short stability continuation.
if isinstance(source_checkpoint, dict) and "optimizer_state_dict" in source_checkpoint:
    try:
        optimizer.load_state_dict(source_checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = LEARNING_RATE
        print("Restored optimizer state; set learning rate to", LEARNING_RATE)
    except Exception as exc:
        print("Optimizer state could not be restored; using a fresh optimizer.")
        print("Reason:", exc)

bce_loss = nn.BCEWithLogitsLoss(reduction="none")


def calculate_loss(logits, targets, valid_slices):
    """Compute masked BCE + Dice in FP32; exclude padded depth slices."""
    logits = logits.float()
    targets = targets.float()
    valid_slices = valid_slices.float()

    check_finite_tensor("logits", logits)
    check_finite_tensor("targets", targets)

    valid_mask = valid_slices[:, None, :, None, None].to(
        device=logits.device, dtype=torch.float32
    )

    valid_voxels = valid_mask.sum() * logits.shape[-2] * logits.shape[-1]
    if valid_voxels.item() <= 0:
        raise ValueError("Batch contains no valid CT slices.")

    bce_map = bce_loss(logits, targets)
    bce = (bce_map * valid_mask).sum() / valid_voxels.clamp_min(1.0)

    probabilities = torch.sigmoid(logits) * valid_mask
    masked_targets = targets * valid_mask

    dims = (1, 2, 3, 4)
    intersection = (probabilities * masked_targets).sum(dim=dims)
    denominator = probabilities.sum(dim=dims) + masked_targets.sum(dim=dims)
    dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)

    loss = bce + (1.0 - dice.mean())
    if not torch.isfinite(loss):
        raise FloatingPointError(
            f"Non-finite loss: BCE={bce.item()}, Dice={dice.detach().cpu().tolist()}"
        )
    return loss


def train_one_epoch():
    model.train()
    total_loss = 0.0
    successful = 0
    skipped = 0

    for batch_index, batch in enumerate(train_loader, start=1):
        images = batch["image"].to(device, non_blocking=True).float()
        masks = batch["mask"].to(device, non_blocking=True).float()
        valid_slices = batch["valid_slices"].to(device, non_blocking=True).float()

        if not torch.isfinite(images).all() or not torch.isfinite(masks).all():
            skipped += 1
            print(f"Invalid input/mask in training batch {batch_index}; skipped.")
            continue

        optimizer.zero_grad(set_to_none=True)

        # FP32 forward + FP32 loss for stability.
        logits = model(images).float()
        try:
            loss = calculate_loss(logits, masks, valid_slices)
        except (ValueError, FloatingPointError) as exc:
            skipped += 1
            print(f"Training batch {batch_index} skipped: {exc}")
            continue

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0, error_if_nonfinite=False
        )
        if not torch.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            skipped += 1
            print(f"Non-finite gradient norm in batch {batch_index}; skipped.")
            continue

        optimizer.step()
        total_loss += float(loss.item())
        successful += 1

        if batch_index % 100 == 0:
            print(
                f"  Batch [{batch_index}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    if successful == 0:
        raise RuntimeError("No training batches completed successfully.")

    print(f"Successful training batches: {successful}; skipped: {skipped}")
    return total_loss / successful


@torch.no_grad()
def validate():
    model.eval()
    total_loss = 0.0
    successful = 0
    skipped = 0
    intersection_total = 0.0
    predicted_total = 0.0
    target_total = 0.0

    for batch_index, batch in enumerate(val_loader, start=1):
        images = batch["image"].to(device, non_blocking=True).float()
        masks = batch["mask"].to(device, non_blocking=True).float()
        valid_slices = batch["valid_slices"].to(device, non_blocking=True).float()

        if not torch.isfinite(images).all() or not torch.isfinite(masks).all():
            skipped += 1
            print(f"Invalid input/mask in validation batch {batch_index}; skipped.")
            continue

        logits = model(images).float()
        try:
            loss = calculate_loss(logits, masks, valid_slices)
        except (ValueError, FloatingPointError) as exc:
            skipped += 1
            print(f"Validation batch {batch_index} skipped: {exc}")
            continue

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= THRESHOLD).float()
        valid_mask = valid_slices[:, None, :, None, None]

        predictions = predictions * valid_mask
        valid_targets = masks * valid_mask

        intersection_total += (predictions * valid_targets).sum().item()
        predicted_total += predictions.sum().item()
        target_total += valid_targets.sum().item()
        total_loss += loss.item()
        successful += 1

    if successful == 0:
        raise RuntimeError(
            "No validation batches completed. Inspect the first invalid batch "
            "and the source image/mask files; do not continue training."
        )

    dice = (2.0 * intersection_total + 1e-8) / (
        predicted_total + target_total + 1e-8
    )
    iou = (intersection_total + 1e-8) / (
        predicted_total + target_total - intersection_total + 1e-8
    )

    print(f"Successful validation batches: {successful}; skipped: {skipped}")
    return total_loss / successful, dice, iou


def save_checkpoint(path, epoch, val_loss, val_dice, val_iou):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": float(val_loss),
            "val_dice": float(val_dice),
            "val_iou": float(val_iou),
            "threshold": THRESHOLD,
            "amp_enabled": False,
            "source_checkpoint": str(SOURCE_CHECKPOINT),
        },
        path,
    )


# Keep a separate copy of the source best checkpoint as the initial best
# candidate for this experiment. Existing files are never overwritten.
best_dice = source_val_dice
if not STABLE_BEST_CHECKPOINT.exists():
    save_checkpoint(
        STABLE_BEST_CHECKPOINT,
        source_epoch,
        float(source_checkpoint.get("val_loss", float("nan")))
        if isinstance(source_checkpoint, dict) else float("nan"),
        source_val_dice,
        source_val_iou,
    )

history = []
if HISTORY_PATH.exists():
    with open(HISTORY_PATH, "r", newline="", encoding="utf-8") as f:
        history = list(csv.DictReader(f))

fieldnames = [
    "epoch", "train_loss", "val_loss", "val_dice",
    "val_iou", "learning_rate", "successful_train_batches",
]

print("\nStarting short FP32 continuation...")
for offset in range(1, ADDITIONAL_EPOCHS + 1):
    epoch = source_epoch + offset
    print(f"\nEpoch [{epoch}] (stable continuation)")

    train_loss = train_one_epoch()
    val_loss, val_dice, val_iou = validate()
    current_lr = optimizer.param_groups[0]["lr"]

    print(f"Train Loss:      {train_loss:.6f}")
    print(f"Validation Loss: {val_loss:.6f}")
    print(f"Validation Dice: {val_dice:.6f}")
    print(f"Validation IoU:  {val_iou:.6f}")
    print(f"Learning Rate:   {current_lr:.8f}")

    history.append({
        "epoch": epoch,
        "train_loss": f"{train_loss:.6f}",
        "val_loss": f"{val_loss:.6f}",
        "val_dice": f"{val_dice:.6f}",
        "val_iou": f"{val_iou:.6f}",
        "learning_rate": f"{current_lr:.8f}",
        "successful_train_batches": "",
    })

    # Save latest after every completed epoch, regardless of whether it
    # improved the best validation Dice.
    save_checkpoint(
        LATEST_CHECKPOINT, epoch, val_loss, val_dice, val_iou
    )

    if val_dice > best_dice:
        best_dice = val_dice
        save_checkpoint(
            STABLE_BEST_CHECKPOINT, epoch, val_loss, val_dice, val_iou
        )
        print(f"New best stable checkpoint saved: {STABLE_BEST_CHECKPOINT}")

    with open(HISTORY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

print("\n" + "=" * 72)
print("STABLE CONTINUATION FINISHED")
print("=" * 72)
print(f"Best validation Dice seen (including source checkpoint): {best_dice:.6f}")
print(f"Stable best checkpoint: {STABLE_BEST_CHECKPOINT}")
print(f"Latest checkpoint: {LATEST_CHECKPOINT}")
print(f"New history: {HISTORY_PATH}")
print("Original best_unet3d.pth and best_unet3d_dropout.pth were not overwritten.")
