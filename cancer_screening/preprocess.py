import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
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

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "cancer_screening"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "cancer_screening"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# FEATURES
# ============================================================

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

TARGET_COLUMN = "Diagnosis"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def split_data(X, y):

    # First split:
    # 70% Train
    # 30% Temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # Second split:
    # Temporary → 50% Validation + 50% Test
    #
    # Therefore:
    # Train      = 70%
    # Validation = 15%
    # Test       = 15%

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


# ============================================================
# FEATURE SCALING
# ============================================================

def scale_features(X_train, X_val, X_test):

    scaler = StandardScaler()

    # IMPORTANT:
    # Fit ONLY on training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform validation/test using same scaler
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler for future inference
    scaler_path = os.path.join(
        MODEL_DIR,
        "scaler.pkl"
    )

    joblib.dump(scaler, scaler_path)

    print(f"Scaler saved to: {scaler_path}")

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler
    )


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

def save_processed_data(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test
):

    train_df = pd.DataFrame(
        X_train,
        columns=FEATURE_COLUMNS
    )

    train_df[TARGET_COLUMN] = y_train.values

    val_df = pd.DataFrame(
        X_val,
        columns=FEATURE_COLUMNS
    )

    val_df[TARGET_COLUMN] = y_val.values

    test_df = pd.DataFrame(
        X_test,
        columns=FEATURE_COLUMNS
    )

    test_df[TARGET_COLUMN] = y_test.values

    train_path = os.path.join(
        RESULTS_DIR,
        "train_processed.csv"
    )

    val_path = os.path.join(
        RESULTS_DIR,
        "validation_processed.csv"
    )

    test_path = os.path.join(
        RESULTS_DIR,
        "test_processed.csv"
    )

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Training data saved:   {train_path}")
    print(f"Validation data saved: {val_path}")
    print(f"Testing data saved:    {test_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CANCER SCREENING PREPROCESSING")
    print("=" * 60)

    # Load
    X, y = load_data()

    print(f"\nTotal samples: {len(X)}")
    print(f"Total features: {X.shape[1]}")

    # Split
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_data(X, y)

    print("\nDataset split:")
    print(f"Training   : {len(X_train)} samples")
    print(f"Validation : {len(X_val)} samples")
    print(f"Testing    : {len(X_test)} samples")

    # Display class distributions
    print("\nClass distribution:")

    print("\nTraining:")
    print(y_train.value_counts())

    print("\nValidation:")
    print(y_val.value_counts())

    print("\nTesting:")
    print(y_test.value_counts())

    # Scaling
    (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler
    ) = scale_features(
        X_train,
        X_val,
        X_test
    )

    print("\nScaled feature shapes:")
    print("Training  :", X_train_scaled.shape)
    print("Validation:", X_val_scaled.shape)
    print("Testing   :", X_test_scaled.shape)

    # Save
    save_processed_data(
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        y_test
    )

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 60)