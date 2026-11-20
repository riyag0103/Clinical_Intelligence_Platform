import os
import joblib
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "cancer_screening"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_cancer_classifier.pkl"
)


# ============================================================
# FEATURE COLUMNS
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


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"\nScaler not found:\n{SCALER_PATH}"
        )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"\nCancer classifier not found:\n{MODEL_PATH}"
        )

    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)

    return scaler, model


# ============================================================
# VALIDATE PATIENT DATA
# ============================================================

def validate_patient_data(patient_data):

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in patient_data
    ]

    if missing_features:

        raise ValueError(
            f"Missing patient features: {missing_features}"
        )


# ============================================================
# CANCER PREDICTION
# ============================================================

def predict_cancer(patient_data):

    # Validate input
    validate_patient_data(patient_data)

    # Load trained model and scaler
    scaler, model = load_model()

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    features = pd.DataFrame(
        [[
            patient_data["Age"],
            patient_data["Gender"],
            patient_data["BMI"],
            patient_data["Smoking"],
            patient_data["GeneticRisk"],
            patient_data["PhysicalActivity"],
            patient_data["AlcoholIntake"],
            patient_data["CancerHistory"]
        ]],
        columns=FEATURE_COLUMNS
    )

    # --------------------------------------------------------
    # Scale input
    # --------------------------------------------------------

    features_scaled = scaler.transform(features)

    # --------------------------------------------------------
    # IMPORTANT:
    # Convert scaled NumPy array back to DataFrame
    # so Random Forest receives feature names.
    # --------------------------------------------------------

    features_scaled = pd.DataFrame(
        features_scaled,
        columns=FEATURE_COLUMNS
    )

    # --------------------------------------------------------
    # Predict probability
    # --------------------------------------------------------

    cancer_probability = model.predict_proba(
        features_scaled
    )[0][1]

    # --------------------------------------------------------
    # Classification threshold
    # --------------------------------------------------------

    prediction = int(
        cancer_probability >= 0.50
    )

    return cancer_probability, prediction


# ============================================================
# DISPLAY RESULT
# ============================================================

def display_result(
    patient_data,
    cancer_probability,
    prediction
):

    probability_percent = cancer_probability * 100

    print()
    print("=" * 65)
    print("             INITIAL CANCER SCREENING")
    print("=" * 65)

    # --------------------------------------------------------
    # Patient information
    # --------------------------------------------------------

    print("\nPatient Information")
    print("-" * 65)

    for feature in FEATURE_COLUMNS:

        print(
            f"{feature:<20}: "
            f"{patient_data[feature]}"
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print("\nPrediction")
    print("-" * 65)

    print(
        f"Cancer Probability : "
        f"{probability_percent:.2f}%"
    )

    if prediction == 1:

        print(
            "Prediction          : CANCER"
        )

        print(
            "\nDecision             : "
            "PROCEED TO CT ANALYSIS"
        )

        print(
            "\nNext Module:"
        )

        print(
            "CT preprocessing → "
            "tumor segmentation → "
            "tumor characterization"
        )

    else:

        print(
            "Prediction          : NON-CANCER"
        )

        print(
            "\nDecision             : "
            "NO CANCER-SPECIFIC CT ANALYSIS"
        )

    print("=" * 65)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Example patient
    # --------------------------------------------------------

    patient = {

        "Age": 58,

        "Gender": 1,

        "BMI": 26.5,

        "Smoking": 1,

        "GeneticRisk": 2,

        "PhysicalActivity": 3.5,

        "AlcoholIntake": 4.1,

        "CancerHistory": 1
    }

    # --------------------------------------------------------
    # Run prediction
    # --------------------------------------------------------

    probability, prediction = predict_cancer(
        patient
    )

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    display_result(
        patient,
        probability,
        prediction
    )