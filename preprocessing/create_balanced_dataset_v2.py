import pandas as pd
from pathlib import Path

# ----------------------------------
# PATHS
# ----------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

metadata_path = (
    PROJECT_ROOT
    / "datasets"
    / "metadata"
    / "full_metadata.csv"
)

output_path = (
    PROJECT_ROOT
    / "datasets"
    / "metadata"
    / "balanced_dataset_v2.csv"
)

# ----------------------------------
# LOAD DATA
# ----------------------------------

df = pd.read_csv(metadata_path)

print(f"\nTotal Samples: {len(df)}")

# ----------------------------------
# SPLIT
# ----------------------------------

tumor_df = df[
    df["empty_mask"] == False
]

healthy_df = df[
    df["empty_mask"] == True
]

print(
    f"Tumor Samples Available: "
    f"{len(tumor_df)}"
)

print(
    f"Healthy Samples Available: "
    f"{len(healthy_df)}"
)

# ----------------------------------
# SAMPLE
# ----------------------------------

tumor_sample = tumor_df.sample(
    n=5000,
    random_state=42
)

healthy_sample = healthy_df.sample(
    n=5000,
    random_state=42
)

balanced_df = pd.concat(
    [
        tumor_sample,
        healthy_sample
    ]
)

balanced_df = balanced_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# ----------------------------------
# SAVE
# ----------------------------------

balanced_df.to_csv(
    output_path,
    index=False
)

print("\nDataset V2 Created!")

print(
    f"Final Samples: "
    f"{len(balanced_df)}"
)

print(
    f"Saved To:\n{output_path}"
)