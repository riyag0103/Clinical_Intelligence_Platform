import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

PROJECT_ROOT = Path(r"D:\Clinical_Project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.unet3d import UNet3D
from data_loader.dataset_3d_patches import CT3DPatchDataset


# Paths
TRAIN_CSV = PROJECT_ROOT / "datasets" / "metadata" / "3d_splits" / "train_3d.csv"
VAL_CSV = PROJECT_ROOT / "datasets" / "metadata" / "3d_splits" / "val_3d.csv"

# Keep this identical to the image_root used by your original training run.
IMAGE_ROOT = PROJECT_ROOT / "datasets" / "processed" / "ct"
MASK_ROOT = PROJECT_ROOT / "datasets" / "processed" / "lidc_consensus_masks"

CHECKPOINT_DIR = PROJECT_ROOT / "saved_models" / "unet3d"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "unet3d" / "dropout"

# Load original weights; save the dropout model separately.
BASE_CHECKPOINT_PATH = CHECKPOINT_DIR / "best_unet3d.pth"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_unet3d_dropout.pth"
HISTORY_PATH = OUTPUT_DIR / "dropout_training_history.csv"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Training settings
EPOCHS = 100
BATCH_SIZE = 1
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 0
PATCH_DEPTH = 8
STRIDE = 4
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"

print("=" * 70)
print("3D U-NET WITH DROPOUT - FINE-TUNING")
print("=" * 70)
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Base checkpoint: {BASE_CHECKPOINT_PATH}")
print(f"New checkpoint: {BEST_MODEL_PATH}")
print(f"Epochs: {EPOCHS}, learning rate: {LEARNING_RATE}")


# Check required files
for required_path in [
    TRAIN_CSV, VAL_CSV, IMAGE_ROOT, MASK_ROOT, BASE_CHECKPOINT_PATH
]:
    if not required_path.exists():
        raise FileNotFoundError(f"Required path not found:\n{required_path}")


# Datasets and dataloaders
print("\nLoading datasets...")
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
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=use_amp,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=use_amp,
)

print(f"Training patches: {len(train_dataset)}")
print(f"Validation patches: {len(val_dataset)}")


# Initialize dropout model from original trained weights.
print("\nInitializing dropout-enabled 3D U-Net...")
model = UNet3D().to(device)

try:
    base_checkpoint = torch.load(
        BASE_CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )
except TypeError:
    base_checkpoint = torch.load(
        BASE_CHECKPOINT_PATH,
        map_location=device,
    )

if isinstance(base_checkpoint, dict) and "model_state_dict" in base_checkpoint:
    base_state_dict = base_checkpoint["model_state_dict"]
elif isinstance(base_checkpoint, dict) and "state_dict" in base_checkpoint:
    base_state_dict = base_checkpoint["state_dict"]
else:
    base_state_dict = base_checkpoint

# Dropout adds no trainable parameters, so state-dict shapes remain compatible.
model.load_state_dict(base_state_dict, strict=True)
print("Original model weights loaded successfully.")
print("Original checkpoint will not be overwritten.")

parameter_count = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {parameter_count:,}")


# Loss and optimizer
# UNet3D.forward() must return raw logits, without sigmoid.
bce_loss = nn.BCEWithLogitsLoss(reduction="none")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)
scaler = GradScaler("cuda", enabled=use_amp)


def calculate_loss(logits, targets, valid_slices):
    """Masked BCE + Dice loss; padded slices are excluded."""
    targets = targets.float()

    valid_mask = valid_slices[:, None, :, None, None].to(
        device=logits.device,
        dtype=logits.dtype,
    )

    bce_map = bce_loss(logits, targets)
    valid_voxels = (
        valid_mask.sum() * logits.shape[-2] * logits.shape[-1]
    )
    bce = (bce_map * valid_mask).sum() / (valid_voxels + 1e-8)

    probabilities = torch.sigmoid(logits) * valid_mask
    masked_targets = targets * valid_mask
    spatial_dims = (1, 2, 3, 4)

    intersection = (
        probabilities * masked_targets
    ).sum(dim=spatial_dims)

    denominator = (
        probabilities.sum(dim=spatial_dims)
        + masked_targets.sum(dim=spatial_dims)
    )

    dice = (
        2.0 * intersection + 1e-6
    ) / (
        denominator + 1e-6
    )

    return bce + (1.0 - dice.mean())


def train_one_epoch():
    model.train()
    running_loss = 0.0
    successful_batches = 0

    for batch_index, batch in enumerate(train_loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        valid_slices = batch["valid_slices"].to(
            device, non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            loss = calculate_loss(logits, masks, valid_slices)

        if not torch.isfinite(loss):
            print(f"Non-finite loss in batch {batch_index}; skipping.")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        successful_batches += 1

        if batch_index % 100 == 0:
            print(
                f"  Batch [{batch_index}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    if successful_batches == 0:
        raise RuntimeError("No training batches completed successfully.")

    return running_loss / successful_batches


@torch.no_grad()
def validate():
    model.eval()
    running_loss = 0.0
    successful_batches = 0

    total_intersection = 0.0
    total_predicted = 0.0
    total_target = 0.0

    for batch in val_loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        valid_slices = batch["valid_slices"].to(
            device, non_blocking=True
        )

        with autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            loss = calculate_loss(logits, masks, valid_slices)

        if not torch.isfinite(loss):
            continue

        running_loss += loss.item()
        successful_batches += 1

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()

        valid_mask = valid_slices[:, None, :, None, None].to(
            device=predictions.device,
            dtype=predictions.dtype,
        )

        predictions = predictions * valid_mask
        masks = masks * valid_mask

        total_intersection += (predictions * masks).sum().item()
        total_predicted += predictions.sum().item()
        total_target += masks.sum().item()

    if successful_batches == 0:
        raise RuntimeError("No validation batches completed successfully.")

    val_loss = running_loss / successful_batches
    val_dice = (
        2.0 * total_intersection + 1e-8
    ) / (
        total_predicted + total_target + 1e-8
    )
    val_iou = (
        total_intersection + 1e-8
    ) / (
        total_predicted + total_target - total_intersection + 1e-8
    )

    return val_loss, val_dice, val_iou


def save_history(history):
    fieldnames = [
        "epoch", "train_loss", "val_loss",
        "val_dice", "val_iou", "learning_rate",
    ]
    with open(HISTORY_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


# Training loop
history = []
best_val_dice = -1.0

print("\nStarting dropout fine-tuning...")
for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch [{epoch}/{EPOCHS}]")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    train_loss = train_one_epoch()
    val_loss, val_dice, val_iou = validate()
    current_lr = optimizer.param_groups[0]["lr"]

    print(f"Train Loss:      {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Dice: {val_dice:.4f}")
    print(f"Validation IoU:  {val_iou:.4f}")
    print(f"Learning Rate:   {current_lr:.7f}")

    history.append({
        "epoch": epoch,
        "train_loss": round(train_loss, 6),
        "val_loss": round(val_loss, 6),
        "val_dice": round(val_dice, 6),
        "val_iou": round(val_iou, 6),
        "learning_rate": current_lr,
    })
    save_history(history)

    if val_dice > best_val_dice:
        best_val_dice = val_dice
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "val_dice": val_dice,
            "val_iou": val_iou,
            "val_loss": val_loss,
            "dropout_encoder3": 0.10,
            "dropout_bottleneck": 0.20,
            "base_checkpoint": str(BASE_CHECKPOINT_PATH),
        }
        torch.save(checkpoint, BEST_MODEL_PATH)
        print(f"Best dropout model saved: {BEST_MODEL_PATH}")

    if device.type == "cuda":
        peak_memory_gb = (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
        )
        print(f"Peak GPU memory: {peak_memory_gb:.2f} GB")

print("\n" + "=" * 70)
print("DROPOUT FINE-TUNING FINISHED")
print("=" * 70)
print(f"Best patch-level validation Dice: {best_val_dice:.4f}")
print(f"Best dropout checkpoint: {BEST_MODEL_PATH}")
print(f"Training history: {HISTORY_PATH}")
print("Original best_unet3d.pth was not overwritten.")