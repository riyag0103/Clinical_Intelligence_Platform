import pandas as pd
import os

# ----------------------------------
# LOAD TRAIN CSV
# ----------------------------------
csv_path = r"datasets/metadata/kfold_splits/fold_1/train.csv"

df = pd.read_csv(csv_path)

# ----------------------------------
# RANDOMLY SAMPLE 1000 RECORDS
# ----------------------------------
mini_df = df.sample(
    n=1000,
    random_state=42
)

# ----------------------------------
# OUTPUT DIRECTORY
# ----------------------------------
output_dir = r"datasets/metadata/mini"

os.makedirs(output_dir, exist_ok=True)

# ----------------------------------
# SAVE MINI DATASET
# ----------------------------------
output_path = os.path.join(
    output_dir,
    "mini_train.csv"
)

mini_df.to_csv(
    output_path,
    index=False
)

print("\nMINI DATASET CREATED!")

print(f"Samples: {len(mini_df)}")