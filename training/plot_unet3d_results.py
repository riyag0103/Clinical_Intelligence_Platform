import csv
from pathlib import Path

import matplotlib.pyplot as plt


# =====================================================
# 1. PATHS
# =====================================================

PROJECT_ROOT = Path(r"D:\Clinical_Project")

HISTORY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "unet3d"
    / "training_history.csv"
)

GRAPH_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "unet3d"
    / "graphs"
)

GRAPH_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. LOAD TRAINING HISTORY
# =====================================================

if not HISTORY_PATH.exists():
    raise FileNotFoundError(
        f"Training history not found:\n{HISTORY_PATH}"
    )

with open(
    HISTORY_PATH,
    "r",
    newline="",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)
    history = list(reader)

if not history:
    raise ValueError(
        "The training history CSV is empty."
    )

required_columns = [
    "epoch",
    "train_loss",
    "val_loss",
    "val_dice",
    "val_iou"
]

missing_columns = [
    column
    for column in required_columns
    if column not in history[0]
]

if missing_columns:
    raise ValueError(
        f"Missing columns in CSV: {missing_columns}\n"
        f"Available columns: {list(history[0].keys())}"
    )


# Convert CSV strings to numerical values.
epochs = [
    int(row["epoch"])
    for row in history
]

train_loss = [
    float(row["train_loss"])
    for row in history
]

val_loss = [
    float(row["val_loss"])
    for row in history
]

val_dice = [
    float(row["val_dice"])
    for row in history
]

val_iou = [
    float(row["val_iou"])
    for row in history
]


# =====================================================
# 3. PRINT RESULTS SUMMARY
# =====================================================

best_dice_index = val_dice.index(max(val_dice))
best_iou_index = val_iou.index(max(val_iou))

print("=" * 55)
print("3D U-NET TRAINING RESULTS")
print("=" * 55)

print(f"Epochs recorded: {len(epochs)}")
print(f"First recorded epoch: {epochs[0]}")
print(f"Last recorded epoch: {epochs[-1]}")

print(
    f"\nLowest training loss: {min(train_loss):.4f}"
)

print(
    f"Lowest validation loss: {min(val_loss):.4f}"
)

print(
    f"Best recorded validation Dice: "
    f"{max(val_dice):.4f} "
    f"at epoch {epochs[best_dice_index]}"
)

print(
    f"Best recorded validation IoU: "
    f"{max(val_iou):.4f} "
    f"at epoch {epochs[best_iou_index]}"
)


# =====================================================
# 4. GRAPH 1: TRAINING LOSS
# =====================================================

plt.figure(figsize=(9, 5))

plt.plot(
    epochs,
    train_loss,
    marker="o",
    label="Training Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("3D U-Net Training Loss")
plt.xticks(epochs)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "training_loss.png",
    dpi=300
)

plt.show()
plt.close()


# =====================================================
# 5. GRAPH 2: VALIDATION LOSS
# =====================================================

plt.figure(figsize=(9, 5))

plt.plot(
    epochs,
    val_loss,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("3D U-Net Validation Loss")
plt.xticks(epochs)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "validation_loss.png",
    dpi=300
)

plt.show()
plt.close()


# =====================================================
# 6. GRAPH 3: VALIDATION DICE SCORE
# =====================================================

plt.figure(figsize=(9, 5))

plt.plot(
    epochs,
    val_dice,
    marker="o",
    label="Validation Dice"
)

plt.axhline(
    y=0.5743,
    linestyle="--",
    label="Previously reported best Dice: 0.5743"
)

plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.title("3D U-Net Validation Dice Score")
plt.xticks(epochs)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "validation_dice.png",
    dpi=300
)

plt.show()
plt.close()


# =====================================================
# 7. GRAPH 4: VALIDATION IOU SCORE
# =====================================================

plt.figure(figsize=(9, 5))

plt.plot(
    epochs,
    val_iou,
    marker="o",
    label="Validation IoU"
)

plt.xlabel("Epoch")
plt.ylabel("IoU Score")
plt.title("3D U-Net Validation IoU Score")
plt.xticks(epochs)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "validation_iou.png",
    dpi=300
)

plt.show()
plt.close()


# =====================================================
# 8. COMBINED SUMMARY GRAPH
# =====================================================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    val_dice,
    marker="o",
    label="Validation Dice"
)

plt.plot(
    epochs,
    val_iou,
    marker="s",
    label="Validation IoU"
)

plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("3D U-Net Segmentation Performance")
plt.xticks(epochs)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "segmentation_performance.png",
    dpi=300
)

plt.show()
plt.close()


# =====================================================
# 9. COMPLETION
# =====================================================

print("\n" + "=" * 55)
print("ALL GRAPHS GENERATED SUCCESSFULLY")
print("=" * 55)

print(f"\nGraphs saved in:\n{GRAPH_DIR}")

for graph_file in sorted(GRAPH_DIR.glob("*.png")):
    print(f"  {graph_file.name}")

print("\nOpen the graphs folder to view your results.")