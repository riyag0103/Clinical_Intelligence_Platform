import os
import sys
import time
import numpy as np
import torch

from torch.utils.data import DataLoader

# ==================================================
# PROJECT ROOT
# ==================================================
PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

sys.path.append(PROJECT_ROOT)

# ==================================================
# IMPORTS
# ==================================================
from data_loader.dataset import CTSegmentationDataset
from models.attention_unet import AttentionUNet
from models.losses import HybridMedicalLoss
from utils.metrics import dice_score, iou_score


def main():

    # ==========================================
    # CONFIG
    # ==========================================
    NUM_FOLDS = 5
    EPOCHS = 15
    BATCH_SIZE = 16

    KFOLD_DIR = os.path.join(
        PROJECT_ROOT,
        "datasets",
        "metadata",
        "kfold_splits"
    )

    BEST_MODEL_DIR = os.path.join(
        PROJECT_ROOT,
        "saved_models",
        "kfold_models"
    )

    os.makedirs(
        BEST_MODEL_DIR,
        exist_ok=True
    )

    # ==========================================
    # DEVICE
    # ==========================================
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\n" + "=" * 60)
    print(f"USING DEVICE: {device}")

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    print("=" * 60)

    # ==========================================
    # RESULTS
    # ==========================================
    fold_dice_scores = []
    fold_iou_scores = []

    # ==========================================
    # K-FOLD LOOP
    # ==========================================
    for fold in range(1, NUM_FOLDS + 1):

        print("\n" + "=" * 60)
        print(f"TRAINING FOLD {fold}")
        print("=" * 60)

        train_csv = os.path.join(
            KFOLD_DIR,
            f"fold_{fold}",
            "train_mini.csv"
        )

        val_csv = os.path.join(
            KFOLD_DIR,
            f"fold_{fold}",
            "val_mini.csv"
        )

        train_dataset = CTSegmentationDataset(
            train_csv
        )

        val_dataset = CTSegmentationDataset(
            val_csv
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        model = AttentionUNet().to(device)

        criterion = HybridMedicalLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-4
        )

        best_dice = 0

        # ======================================
        # TRAINING
        # ======================================
        for epoch in range(EPOCHS):

            epoch_start = time.time()

            model.train()

            running_loss = 0.0

            for images, masks in train_loader:

                images = images.to(
                    device,
                    non_blocking=True
                )

                masks = masks.to(
                    device,
                    non_blocking=True
                )

                predictions = model(images)

                loss = criterion(
                    predictions,
                    masks
                )

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                running_loss += loss.item()

            avg_loss = (
                running_loss /
                len(train_loader)
            )

            epoch_time = (
                time.time()
                - epoch_start
            )

            print(
                f"Fold {fold} | "
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"Loss: {avg_loss:.4f} | "
                f"Time: {epoch_time:.2f}s"
            )

        # ======================================
        # VALIDATION
        # ======================================
        model.eval()

        val_dice = []
        val_iou = []

        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(device)
                masks = masks.to(device)

                predictions = model(images)

                predictions = (
                    predictions > 0.35
                ).float()

                dice = dice_score(
                    predictions,
                    masks
                )

                iou = iou_score(
                    predictions,
                    masks
                )

                val_dice.append(dice)
                val_iou.append(iou)

        avg_dice = np.mean(val_dice)
        avg_iou = np.mean(val_iou)

        fold_dice_scores.append(
            avg_dice
        )

        fold_iou_scores.append(
            avg_iou
        )

        print("\n" + "-" * 50)
        print(f"FOLD {fold} RESULTS")
        print("-" * 50)

        print(f"Dice: {avg_dice:.4f}")
        print(f"IoU : {avg_iou:.4f}")

        if avg_dice > best_dice:

            best_dice = avg_dice

            torch.save(
                model.state_dict(),
                os.path.join(
                    BEST_MODEL_DIR,
                    f"attention_unet_fold_{fold}.pth"
                )
            )

            print(
                f"Best Fold {fold} Model Saved!"
            )

    # ==========================================
    # FINAL REPORT
    # ==========================================
    print("\n" + "=" * 60)
    print("FINAL K-FOLD RESULTS")
    print("=" * 60)

    for i in range(NUM_FOLDS):

        print(
            f"Fold {i+1}: "
            f"Dice={fold_dice_scores[i]:.4f} "
            f"IoU={fold_iou_scores[i]:.4f}"
        )

    print("\n" + "-" * 60)

    print(
        f"Mean Dice: "
        f"{np.mean(fold_dice_scores):.4f}"
    )

    print(
        f"Std Dice: "
        f"{np.std(fold_dice_scores):.4f}"
    )

    print(
        f"Mean IoU: "
        f"{np.mean(fold_iou_scores):.4f}"
    )

    print(
        f"Std IoU: "
        f"{np.std(fold_iou_scores):.4f}"
    )

    print("\nK-FOLD TRAINING COMPLETE!")


if __name__ == "__main__":
    main()