import os
import pandas as pd

from sklearn.model_selection import GroupKFold

# ----------------------------------
# CONFIGURATION
# ----------------------------------
N_SPLITS = 5

metadata_path = (
    r"datasets/metadata/full_metadata.csv"
)

output_dir = (
    r"datasets/metadata/kfold_splits"
)

os.makedirs(
    output_dir,
    exist_ok=True
)

# ----------------------------------
# LOAD DATA
# ----------------------------------
df = pd.read_csv(metadata_path)

print("\nTOTAL SAMPLES:")
print(len(df))

# ----------------------------------
# CHECK REQUIRED COLUMN
# ----------------------------------
if "patient_id" not in df.columns:

    raise Exception(
        "\npatient_id column not found!"
    )

# ----------------------------------
# GROUP K-FOLD
# ----------------------------------
gkf = GroupKFold(
    n_splits=N_SPLITS
)

groups = df["patient_id"]

# ----------------------------------
# CREATE FOLDS
# ----------------------------------
for fold, (train_idx, val_idx) in enumerate(
    gkf.split(
        df,
        groups=groups
    )
):

    fold_number = fold + 1

    print(
        f"\nCreating Fold {fold_number}"
    )

    train_df = df.iloc[
        train_idx
    ].reset_index(drop=True)

    val_df = df.iloc[
        val_idx
    ].reset_index(drop=True)

    # ------------------------------
    # SAVE DIRECTORY
    # ------------------------------
    fold_dir = os.path.join(
        output_dir,
        f"fold_{fold_number}"
    )

    os.makedirs(
        fold_dir,
        exist_ok=True
    )

    train_path = os.path.join(
        fold_dir,
        "train.csv"
    )

    val_path = os.path.join(
        fold_dir,
        "val.csv"
    )

    train_df.to_csv(
        train_path,
        index=False
    )

    val_df.to_csv(
        val_path,
        index=False
    )

    print(
        f"Train Samples: {len(train_df)}"
    )

    print(
        f"Validation Samples: {len(val_df)}"
    )

# ----------------------------------
# FINISHED
# ----------------------------------
print("\n" + "=" * 50)

print(
    "\n5-FOLD GROUP K-FOLD SPLITS CREATED!"
)

print(
    f"\nSaved To:\n{output_dir}"
)

print("\n" + "=" * 50)