import pandas as pd
import os

# ----------------------------------
# INPUT METADATA
# ----------------------------------
metadata_path = r"datasets/metadata/full_metadata.csv"

# ----------------------------------
# OUTPUT DIRECTORY
# ----------------------------------
output_dir = r"datasets/metadata/balanced"

os.makedirs(
    output_dir,
    exist_ok=True
)

# ----------------------------------
# LOAD METADATA
# ----------------------------------
df = pd.read_csv(metadata_path)

print("\nTOTAL SAMPLES:")
print(len(df))

# ----------------------------------
# POSITIVE SAMPLES
# ----------------------------------
positive_df = df[
    df["empty_mask"] == False
]

# ----------------------------------
# NEGATIVE SAMPLES
# ----------------------------------
negative_df = df[
    df["empty_mask"] == True
]

print("\nPOSITIVE SAMPLES:")
print(len(positive_df))

print("\nNEGATIVE SAMPLES:")
print(len(negative_df))

# ----------------------------------
# SAMPLE BALANCED DATA
# ----------------------------------
positive_sampled = positive_df.sample(
    n=min(1000, len(positive_df)),
    random_state=42
)

negative_sampled = negative_df.sample(
    n=min(1000, len(negative_df)),
    random_state=42
)

# ----------------------------------
# COMBINE
# ----------------------------------
balanced_df = pd.concat(
    [
        positive_sampled,
        negative_sampled
    ]
)

# ----------------------------------
# SHUFFLE
# ----------------------------------
balanced_df = balanced_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# ----------------------------------
# SAVE
# ----------------------------------
output_path = os.path.join(
    output_dir,
    "balanced_metadata.csv"
)

balanced_df.to_csv(
    output_path,
    index=False
)

# ----------------------------------
# REPORT
# ----------------------------------
print("\nBALANCED DATASET CREATED!")

print(f"\nFINAL SAMPLES: {len(balanced_df)}")

print(f"\nSaved To:\n{output_path}")