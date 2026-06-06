import os
import pandas as pd

# ----------------------------------
# CONFIG
# ----------------------------------
KFOLD_DIR = r"datasets/metadata/kfold_splits"

TRAIN_SIZE = 5000
VAL_SIZE = 1000

# ----------------------------------
# PROCESS ALL FOLDS
# ----------------------------------
for fold in range(1, 6):

    print(f"\nProcessing Fold {fold}")

    fold_dir = os.path.join(
        KFOLD_DIR,
        f"fold_{fold}"
    )

    train_csv = os.path.join(
        fold_dir,
        "train.csv"
    )

    val_csv = os.path.join(
        fold_dir,
        "val.csv"
    )

    # ----------------------------------
    # LOAD
    # ----------------------------------
    train_df = pd.read_csv(train_csv)

    val_df = pd.read_csv(val_csv)

    # ----------------------------------
    # SAMPLE
    # ----------------------------------
    train_mini = train_df.sample(
        n=min(TRAIN_SIZE, len(train_df)),
        random_state=42
    )

    val_mini = val_df.sample(
        n=min(VAL_SIZE, len(val_df)),
        random_state=42
    )

    # ----------------------------------
    # SAVE
    # ----------------------------------
    train_mini_path = os.path.join(
        fold_dir,
        "train_mini.csv"
    )

    val_mini_path = os.path.join(
        fold_dir,
        "val_mini.csv"
    )

    train_mini.to_csv(
        train_mini_path,
        index=False
    )

    val_mini.to_csv(
        val_mini_path,
        index=False
    )

    print(
        f"Train Mini: {len(train_mini)}"
    )

    print(
        f"Val Mini: {len(val_mini)}"
    )

print("\n" + "=" * 50)

print(
    "\nMINI K-FOLD SPLITS CREATED!"
)

print("\n" + "=" * 50)