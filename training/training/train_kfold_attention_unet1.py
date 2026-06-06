import pandas as pd

train_df = pd.read_csv(
    r"datasets/metadata/kfold_splits/fold_1/train.csv"
)

val_df = pd.read_csv(
    r"datasets/metadata/kfold_splits/fold_1/val.csv"
)

train_patients = set(train_df["patient_id"])
val_patients = set(val_df["patient_id"])

overlap = train_patients.intersection(
    val_patients
)

print(
    f"Overlapping Patients: {len(overlap)}"
)

if len(overlap) == 0:
    print(
        "\nSUCCESS: No patient leakage detected."
    )
else:
    print(
        "\nWARNING: Leakage detected."
    )