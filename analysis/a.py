import pandas as pd

df = pd.read_csv(
    r"datasets/metadata/kfold_splits/fold_1/train.csv"
)

print(df.head())