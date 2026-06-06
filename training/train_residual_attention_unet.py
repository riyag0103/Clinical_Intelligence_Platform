import sys
import os

# ----------------------------------
# ADD PROJECT ROOT
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
from data_loader.dataset import (
    CTSegmentationDataset
)

from models.residual_attention_unet import (
    ResidualAttentionUNet
)

from models.losses import (
    HybridMedicalLoss
)

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
# LOAD DATASET
# ----------------------------------
metadata_path = (
    r"datasets/metadata/balanced/balanced_metadata.csv"
)

dataset = CTSegmentationDataset(metadata_path)

# ----------------------------------
# DATALOADER
# ----------------------------------
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

# ----------------------------------
# MODEL
# ----------------------------------
model = ResidualAttentionUNet().to(device)

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
# TRAINING LOOP
# ----------------------------------
for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    running_dice = 0.0

    running_iou = 0.0

    for batch_idx, (images, masks) in enumerate(dataloader):

        images = images.to(device)

        masks = masks.to(device)

        # ----------------------------------
        # FORWARD
        # ----------------------------------
        predictions = model(images)

        loss = criterion(
            predictions,
            masks
        )

        # ----------------------------------
        # BACKPROP
        # ----------------------------------
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # ----------------------------------
        # METRICS
        # ----------------------------------
        pred_binary = (
            predictions > 0.35
        ).float()

        dice = dice_score(
            pred_binary,
            masks
        )

        iou = iou_score(
            pred_binary,
            masks
        )

        running_loss += loss.item()

        running_dice += dice

        running_iou += iou

        # ----------------------------------
        # LOGGING
        # ----------------------------------
        if batch_idx % 50 == 0:

            print(
                f"Epoch [{epoch+1}/{EPOCHS}] "
                f"Batch [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f}"
            )

    # ----------------------------------
    # EPOCH METRICS
    # ----------------------------------
    avg_loss = running_loss / len(dataloader)

    avg_dice = running_dice / len(dataloader)

    avg_iou = running_iou / len(dataloader)

    print("\n" + "=" * 60)

    print(f"\nEpoch [{epoch+1}/{EPOCHS}] Completed")

    print(f"\nAverage Train Loss: {avg_loss:.4f}")

    print(f"\nValidation Dice Score: {avg_dice:.4f}")

    print(f"\nValidation IoU Score: {avg_iou:.4f}")

    print("\n" + "=" * 60)

    # ----------------------------------
    # SAVE BEST MODEL
    # ----------------------------------
    if avg_dice > best_dice:

        best_dice = avg_dice

        os.makedirs(
            "outputs/checkpoints",
            exist_ok=True
        )

        torch.save(
            model.state_dict(),
            "outputs/checkpoints/best_residual_attention_unet.pth"
        )

        print("\nBEST RESIDUAL ATTENTION U-NET SAVED!")

# ----------------------------------
# TRAINING COMPLETE
# ----------------------------------
print("\nRESIDUAL ATTENTION U-NET TRAINING COMPLETED!")