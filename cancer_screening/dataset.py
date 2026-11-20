import os
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "cancer_screening",
    "The_Cancer_data_1500_V2.csv"
)

TARGET_COLUMN = "Diagnosis"

FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "BMI",
    "Smoking",
    "GeneticRisk",
    "PhysicalActivity",
    "AlcoholIntake",
    "CancerHistory"
]


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("CANCER SCREENING DATASET")
    print("=" * 60)

    print(f"Dataset path : {DATA_PATH}")
    print(f"Samples      : {len(df)}")
    print(f"Features     : {len(FEATURE_COLUMNS)}")
    print()

    print("Columns:")
    print(df.columns.tolist())
    print()

    return df


# ============================================================
# VALIDATE DATASET
# ============================================================

def validate_dataset(df):

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Check missing values
    missing_values = df[required_columns].isnull().sum()

    print("Missing values:")
    print(missing_values)
    print()

    # Check target
    print("Target distribution:")
    print(df[TARGET_COLUMN].value_counts())
    print()

    print("Target percentages:")
    print(
        df[TARGET_COLUMN]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print()

    return True


# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================

def get_features_and_target(df):

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    dataframe = load_dataset()

    validate_dataset(dataframe)

    X, y = get_features_and_target(dataframe)

    print("=" * 60)
    print("FEATURE MATRIX")
    print("=" * 60)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print()
    print("Feature columns:")
    print(X.columns.tolist())

    print()
    print("First 5 samples:")
    print(X.head())

    print()
    print("First 5 targets:")
    print(y.head())

    print()
    print("Dataset loaded successfully!")