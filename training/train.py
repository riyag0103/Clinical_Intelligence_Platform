import sys
import os

# ----------------------------------
# ADD PROJECT ROOT TO PYTHON PATH
# ----------------------------------
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import torch
from torch.utils.data import DataLoader

from data_loader.dataset import CTSegmentationDataset

from data_loader.augmentations import (
    train_transforms,
    val_transforms
)

# ----------------------------------
# IMPORT ATTENTION U-NET
# ----------------------------------
from models.attention_unet import AttentionUNet

from models.losses import HybridMedicalLoss

from utils.metrics import (
    dice_score,
    iou_score
)

# ----------------------------------
# DEVICE
# ----------------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"\nUSING DEVICE: {device}")

# ----------------------------------
# MINI DATASET PATHS
# ----------------------------------
train_csv = r"datasets/metadata/balanced/balanced_metadata.csv"

val_csv = r"datasets/metadata/kfold_splits/fold_1/val.csv"

# ----------------------------------
# DATASETS
# ----------------------------------
train_dataset = CTSegmentationDataset(
    train_csv,
    transforms=train_transforms
)

val_dataset = CTSegmentationDataset(
    val_csv,
    transforms=val_transforms
)

# ----------------------------------
# DATALOADERS
# ----------------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

# ----------------------------------
# LOAD MODEL
# ----------------------------------
model = AttentionUNet().to(device)

print("\nATTENTION U-NET LOADED SUCCESSFULLY!")

# ----------------------------------
# LOSS FUNCTION
# ----------------------------------
criterion = HybridMedicalLoss()

# ----------------------------------
# OPTIMIZER
# ----------------------------------
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

# ----------------------------------
# TRAINING SETTINGS
# ----------------------------------
EPOCHS = 5

best_dice = 0.0

# ----------------------------------
# CREATE CHECKPOINT DIRECTORY
# ----------------------------------
os.makedirs(
    "outputs/checkpoints",
    exist_ok=True
)

# ----------------------------------
# TRAINING LOOP
# ----------------------------------
for epoch in range(EPOCHS):

    # ----------------------------------
    # TRAIN MODE
    # ----------------------------------
    model.train()

    train_loss = 0.0

    # ----------------------------------
    # TRAINING
    # ----------------------------------
    for batch_idx, (images, masks) in enumerate(train_loader):

        images = images.to(device)

        masks = masks.to(device)

        # ----------------------------------
        # FORWARD PASS
        # ----------------------------------
        predictions = model(images)

        # ----------------------------------
        # LOSS
        # ----------------------------------
        loss = criterion(
            predictions,
            masks
        )

        # ----------------------------------
        # BACKPROPAGATION
        # ----------------------------------
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        # ----------------------------------
        # PRINT PROGRESS
        # ----------------------------------
        if batch_idx % 20 == 0:

            print(
                f"Epoch [{epoch+1}/{EPOCHS}] "
                f"Batch [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    # ----------------------------------
    # VALIDATION
    # ----------------------------------
    model.eval()

    val_dice = 0.0
    val_iou = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)

            masks = masks.to(device)

            predictions = model(images)

            # ----------------------------------
            # THRESHOLD PREDICTIONS
            # ----------------------------------
            predictions = (
                predictions > 0.5
            ).float()

            # ----------------------------------
            # CALCULATE METRICS
            # ----------------------------------
            val_dice += dice_score(
                predictions,
                masks
            )

            val_iou += iou_score(
                predictions,
                masks
            )

    # ----------------------------------
    # AVERAGE METRICS
    # ----------------------------------
    avg_train_loss = (
        train_loss / len(train_loader)
    )

    avg_dice = (
        val_dice / len(val_loader)
    )

    avg_iou = (
        val_iou / len(val_loader)
    )

    # ----------------------------------
    # PRINT RESULTS
    # ----------------------------------
    print("\n" + "=" * 50)

    print(f"Epoch [{epoch+1}/{EPOCHS}] Completed")

    print(f"Average Train Loss: {avg_train_loss:.4f}")

    print(f"Validation Dice Score: {avg_dice:.4f}")

    print(f"Validation IoU Score: {avg_iou:.4f}")

    print("=" * 50)

    # ----------------------------------
    # SAVE BEST MODEL
    # ----------------------------------
    if avg_dice > best_dice:

        best_dice = avg_dice

        torch.save(
            model.state_dict(),
            "outputs/checkpoints/best_attention_unet.pth"
        )

        print("\nBEST ATTENTION U-NET SAVED!\n")

print("\nATTENTION U-NET TRAINING COMPLETED!")