import joblib
import pandas as pd

from backend.config import CANCER_MODEL_PATH, SCALER_PATH, FEATURE_COLUMNS


class CancerClassifier:
    def __init__(self):
        self.scaler = None
        self.model = None

    @property
    def is_loaded(self):
        return self.scaler is not None and self.model is not None

    def load(self):
        if not SCALER_PATH.is_file():
            raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")
        if not CANCER_MODEL_PATH.is_file():
            raise FileNotFoundError(f"GridSearchCV classifier not found: {CANCER_MODEL_PATH}")

        self.scaler = joblib.load(SCALER_PATH)
        self.model = joblib.load(CANCER_MODEL_PATH)

        if hasattr(self.model, "n_features_in_") and self.model.n_features_in_ != len(FEATURE_COLUMNS):
            raise ValueError("Classifier feature count does not match FEATURE_COLUMNS.")

    def predict(self, patient_data: dict):
        if not self.is_loaded:
            raise RuntimeError("Cancer classifier has not been loaded.")

        missing = [name for name in FEATURE_COLUMNS if name not in patient_data]
        if missing:
            raise ValueError(f"Missing patient features: {missing}")

        frame = pd.DataFrame(
            [[float(patient_data[name]) for name in FEATURE_COLUMNS]],
            columns=FEATURE_COLUMNS,
        )
        scaled = self.scaler.transform(frame)
        scaled_frame = pd.DataFrame(scaled, columns=FEATURE_COLUMNS)

        probability = float(self.model.predict_proba(scaled_frame)[0, 1])
        prediction = int(probability >= 0.50)

        return {
            "model": "GridSearchCV Gradient Boosting",
            "cancer_risk_probability": round(probability, 6),
            "cancer_risk_percentage": round(probability * 100, 2),
            "predicted_class": prediction,
            "class_label": "higher_model_estimated_risk" if prediction else "lower_model_estimated_risk",
            "threshold": 0.50,
            "interpretation": (
                "Model-estimated risk signal only; this is not a cancer diagnosis "
                "and must not be used alone to decide whether CT imaging is needed."
            ),
        }
