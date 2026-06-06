import pandas as pd
import os
from sklearn.model_selection import GroupKFold

# ----------------------------------
# LOAD CLEAN METADATA
# ----------------------------------
csv_path = r"datasets/metadata/clean_ct_metadata.csv"

df = pd.read_csv(csv_path)

# ----------------------------------
# OUTPUT DIRECTORY
# ----------------------------------
output_dir = r"datasets/metadata/kfold_splits"

os.makedirs(output_dir, exist_ok=True)

# ----------------------------------
# GROUP K-FOLD
# ----------------------------------
NUM_FOLDS = 5

gkf = GroupKFold(n_splits=NUM_FOLDS)

# ----------------------------------
# PATIENT GROUPS
# ----------------------------------
groups = df["patient_id"]

# ----------------------------------
# CREATE FOLDS
# ----------------------------------
for fold, (train_idx, val_idx) in enumerate(
    gkf.split(df, groups=groups)
):

    print(f"\nProcessing Fold {fold+1}")

    # ----------------------------------
    # SPLIT DATA
    # ----------------------------------
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    # ----------------------------------
    # CREATE FOLD DIRECTORY
    # ----------------------------------
    fold_dir = os.path.join(
        output_dir,
        f"fold_{fold+1}"
    )

    os.makedirs(fold_dir, exist_ok=True)

    # ----------------------------------
    # SAVE CSV FILES
    # ----------------------------------
    train_df.to_csv(
        os.path.join(fold_dir, "train.csv"),
        index=False
    )

    val_df.to_csv(
        os.path.join(fold_dir, "val.csv"),
        index=False
    )

    # ----------------------------------
    # PRINT STATISTICS
    # ----------------------------------
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

print("\nGROUP K-FOLD SPLITTING COMPLETED!")