import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from cancer_screening.model import get_models


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "cancer_screening"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "cancer_screening"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD PROCESSED DATA
# ============================================================

def load_processed_data():

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

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["Diagnosis"])
    y_train = train_df["Diagnosis"]

    X_val = val_df.drop(columns=["Diagnosis"])
    y_val = val_df["Diagnosis"]

    X_test = test_df.drop(columns=["Diagnosis"])
    y_test = test_df["Diagnosis"]

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X, y):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, predictions)

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y,
        probabilities
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("INITIAL CANCER SCREENING MODEL TRAINING")
    print("=" * 70)

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) = load_processed_data()

    print("\nDataset sizes:")
    print("Training   :", len(X_train))
    print("Validation :", len(X_val))
    print("Testing    :", len(X_test))

    models = get_models()

    validation_results = {}

    # ========================================================
    # TRAIN AND VALIDATE
    # ========================================================

    for name, model in models.items():

        print("\n" + "=" * 70)
        print(f"TRAINING: {name}")
        print("=" * 70)

        model.fit(
            X_train,
            y_train
        )

        results = evaluate_model(
            model,
            X_val,
            y_val
        )

        validation_results[name] = results

        print("\nValidation Results:")
        print(
            f"Accuracy  : {results['accuracy']:.4f}"
        )
        print(
            f"Precision : {results['precision']:.4f}"
        )
        print(
            f"Recall    : {results['recall']:.4f}"
        )
        print(
            f"F1 Score  : {results['f1']:.4f}"
        )
        print(
            f"ROC-AUC   : {results['roc_auc']:.4f}"
        )

        print("\nConfusion Matrix:")
        print(results["confusion_matrix"])

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    comparison = []

    for name, result in validation_results.items():

        comparison.append({
            "Model": name,
            "Accuracy": result["accuracy"],
            "Precision": result["precision"],
            "Recall": result["recall"],
            "F1": result["f1"],
            "ROC-AUC": result["roc_auc"]
        })

    comparison_df = pd.DataFrame(comparison)

    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    comparison_path = os.path.join(
        RESULTS_DIR,
        "model_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False
    )

    print(
        f"\nComparison saved to:\n{comparison_path}"
    )

    # ========================================================
    # SELECT BEST MODEL
    # ========================================================

    # Primary criterion: validation ROC-AUC
    best_model_name = max(
        validation_results,
        key=lambda name:
        validation_results[name]["roc_auc"]
    )

    best_model = models[best_model_name]

    print("\n" + "=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(
        f"Selected model: {best_model_name}"
    )

    # ========================================================
    # RETRAIN BEST MODEL ON TRAIN + VALIDATION
    # ========================================================

    X_train_full = pd.concat(
        [X_train, X_val],
        axis=0
    )

    y_train_full = pd.concat(
        [y_train, y_val],
        axis=0
    )

    print("\nRetraining best model using:")
    print(
        f"{len(X_train_full)} training samples"
    )

    best_model.fit(
        X_train_full,
        y_train_full
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    test_results = evaluate_model(
        best_model,
        X_test,
        y_test
    )

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(
        f"Accuracy  : {test_results['accuracy']:.4f}"
    )
    print(
        f"Precision : {test_results['precision']:.4f}"
    )
    print(
        f"Recall    : {test_results['recall']:.4f}"
    )
    print(
        f"F1 Score  : {test_results['f1']:.4f}"
    )
    print(
        f"ROC-AUC   : {test_results['roc_auc']:.4f}"
    )

    print("\nConfusion Matrix:")
    print(test_results["confusion_matrix"])

    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    model_path = os.path.join(
        MODEL_DIR,
        "best_cancer_classifier.pkl"
    )

    joblib.dump(
        best_model,
        model_path
    )

    print(
        f"\nBest model saved to:\n{model_path}"
    )

    print("\n" + "=" * 70)
    print("CANCER SCREENING TRAINING COMPLETED")
    print("=" * 70)