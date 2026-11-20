from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "saved_models" / "cancer_screening"
CANCER_MODEL_PATH = MODEL_DIR / "best_cancer_classifier_gridsearch.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
UNET3D_CHECKPOINT_PATH = PROJECT_ROOT / "saved_models" / "unet3d" / "best_unet3d_dropout.pth"
OUTPUT_DIR = PROJECT_ROOT / "backend" / "outputs"

PATCH_DEPTH = 8
PATCH_STRIDE = 4
IMAGE_SIZE = 128
DEFAULT_THRESHOLD = 0.20

FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "BMI",
    "Smoking",
    "GeneticRisk",
    "PhysicalActivity",
    "AlcoholIntake",
    "CancerHistory",
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
