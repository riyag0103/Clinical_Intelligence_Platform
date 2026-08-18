import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    NUM_FOLDS = 1
    EPOCHS = 1
    BATCH_SIZE = 16
    THRESHOLD = 0.35

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

    RESULTS_DIR = os.path.join(
        PROJECT_ROOT,
        "results"
    )

    os.makedirs(
        BEST_MODEL_DIR,
        exist_ok=True
    )

    os.makedirs(
        RESULTS_DIR,
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
            "train.csv"
        )

        val_csv = os.path.join(
            KFOLD_DIR,
            f"fold_{fold}",
            "val.csv"
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
        best_iou = 0

        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_dice": [],
            "val_dice": [],
            "train_iou": [],
            "val_iou": []
        }

        # ======================================
        # TRAINING
        # ======================================
        for epoch in range(EPOCHS):

            epoch_start = time.time()

            model.train()

            running_loss = 0.0
            train_dice_scores = []
            train_iou_scores = []

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

                with torch.no_grad():

                    binary_preds = (
                        predictions > THRESHOLD
                    ).float()

                    train_dice_scores.append(
                        dice_score(
                            binary_preds,
                            masks
                        )
                    )

                    train_iou_scores.append(
                        iou_score(
                            binary_preds,
                            masks
                        )
                    )

            avg_loss = (
                running_loss /
                len(train_loader)
            )

            avg_train_dice = np.mean(
                train_dice_scores
            )

            avg_train_iou = np.mean(
                train_iou_scores
            )

            # ==================================
            # VALIDATION EVERY EPOCH
            # ==================================
            model.eval()

            val_loss = 0.0
            val_dice_scores = []
            val_iou_scores = []

            with torch.no_grad():

              for images, masks in val_loader:

                images = images.to(device)
                masks = masks.to(device)

                predictions = model(images)

                # DEBUG
                print("\n========== DEBUG ==========")
                print("Predictions shape:", predictions.shape)
                print("Masks shape:", masks.shape)

                print(
                    "Prediction min/max:",
                     predictions.min().item(),
                     predictions.max().item()
                )       

                print(
                    "Mask unique values:",
                     torch.unique(masks)
                )

                print("===========================\n")

                loss = criterion(
                    predictions,
                    masks
                )

                val_loss += loss.item()

                binary_preds = (
                    predictions > THRESHOLD
                ).float()

                val_dice_scores.append(
                    dice_score(
                        binary_preds,
                        masks
                    )
                )

                val_iou_scores.append(
                    iou_score(
                        binary_preds,
                        masks
                    )
                )

            avg_val_loss = (
                val_loss /
                len(val_loader)
            )

            avg_val_dice = np.mean(
                val_dice_scores
            )

            avg_val_iou = np.mean(
                val_iou_scores
            )

            # ==================================
            # SAVE HISTORY
            # ==================================
            history["epoch"].append(
                epoch + 1
            )

            history["train_loss"].append(
                avg_loss
            )

            history["val_loss"].append(
                avg_val_loss
            )

            history["train_dice"].append(
                avg_train_dice
            )

            history["val_dice"].append(
                avg_val_dice
            )

            history["train_iou"].append(
                avg_train_iou
            )

            history["val_iou"].append(
                avg_val_iou
            )

            epoch_time = (
                time.time()
                - epoch_start
            )

            print(
                f"Fold {fold} | "
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"Train Loss={avg_loss:.4f} | "
                f"Val Loss={avg_val_loss:.4f} | "
                f"Train Dice={avg_train_dice:.4f} | "
                f"Val Dice={avg_val_dice:.4f}"
            )

            # Save best model
            if avg_val_dice > best_dice:

                 best_dice = avg_val_dice
                 best_iou = avg_val_iou

            torch.save(
                model.state_dict(),
                os.path.join(
                 BEST_MODEL_DIR,
                 f"attention_unet_fold_{fold}.pth"
                )
            )

        # ======================================
        # FINAL FOLD RESULTS
        # ======================================
        avg_dice = best_dice
        avg_iou = best_iou

        fold_dice_scores.append(
            avg_dice
        )

        fold_iou_scores.append(
            avg_iou
        )

        # ======================================
        # SAVE CSV
        # ======================================
        history_df = pd.DataFrame(
            history
        )

        history_csv = os.path.join(
            RESULTS_DIR,
            f"fold_{fold}_history.csv"
        )

        history_df.to_csv(
            history_csv,
            index=False
        )

        # ======================================
        # GENERATE CURVES
        # ======================================
        plt.figure(
            figsize=(15, 10)
        )

        # LOSS
        plt.subplot(2, 2, 1)

        plt.plot(
            history["train_loss"],
            label="Train"
        )

        plt.plot(
            history["val_loss"],
            label="Val"
        )

        plt.title("Loss")
        plt.legend()

        # DICE
        plt.subplot(2, 2, 2)

        plt.plot(
            history["train_dice"],
            label="Train"
        )

        plt.plot(
            history["val_dice"],
            label="Val"
        )

        plt.title("Dice")
        plt.legend()

        # IOU
        plt.subplot(2, 2, 3)

        plt.plot(
            history["train_iou"],
            label="Train"
        )

        plt.plot(
            history["val_iou"],
            label="Val"
        )

        plt.title("IoU")
        plt.legend()

        # OVERFITTING GAP
        gap = np.array(
            history["train_dice"]
        ) - np.array(
            history["val_dice"]
        )

        plt.subplot(2, 2, 4)

        plt.plot(
            gap,
            label="Gap"
        )

        plt.title(
            "Overfitting Gap"
        )

        plt.legend()

        plt.tight_layout()

        graph_path = os.path.join(
            RESULTS_DIR,
            f"fold_{fold}_curves.png"
        )

        plt.savefig(
            graph_path
        )

        plt.close()

        print("\n" + "-" * 50)
        print(f"FOLD {fold} RESULTS")
        print("-" * 50)

        print(f"Dice: {avg_dice:.4f}")
        print(f"IoU : {avg_iou:.4f}")

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