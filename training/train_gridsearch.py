import os
import sys
import joblib
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay,
    ConfusionMatrixDisplay
)

from cancer_screening.model import get_models

# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

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

PLOT_DIR = os.path.join(
    RESULTS_DIR,
    "plots"
)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD PROCESSED DATA
# ============================================================

def load_data():

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

    # Check feature consistency
    if list(X_train.columns) != list(X_val.columns):
        raise ValueError("Training and validation features do not match.")

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Training and testing features do not match.")

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )


# ============================================================
# 3. HYPERPARAMETER GRIDS
# ============================================================

PARAM_GRIDS = {

    "Logistic Regression": {

        "C": [0.01, 0.1, 1, 10, 100],

        "solver": ["lbfgs"],

        "class_weight": ["balanced", None]
    },

    "Random Forest": {

        "n_estimators": [100, 200, 300],

        "max_depth": [None, 5, 10, 20],

        "min_samples_split": [2, 5, 10],

        "min_samples_leaf": [1, 2, 4]
    },

    "Gradient Boosting": {

        "n_estimators": [100, 200, 300],

        "learning_rate": [0.01, 0.05, 0.1],

        "max_depth": [2, 3, 5],

        "min_samples_split": [2, 5],

        "min_samples_leaf": [1, 2]
    }
}


# ============================================================
# 4. MODEL EVALUATION
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
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "Confusion Matrix": cm
    }


# ============================================================
# 5. MAIN TRAINING PIPELINE
# ============================================================

if __name__ == "__main__":

    print("=" * 75)
    print("CANCER SCREENING - GRIDSEARCHCV")
    print("=" * 75)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) = load_data()

    print("\nDATASET INFORMATION")

    print("Training samples  :", len(X_train))
    print("Validation samples:", len(X_val))
    print("Testing samples   :", len(X_test))

    print("\nFeatures:", list(X_train.columns))

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    models = get_models()

    # --------------------------------------------------------
    # CROSS VALIDATION
    # --------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    tuned_models = {}

    validation_results = []

    # ========================================================
    # 6. GRID SEARCH FOR EACH CLASSIFIER
    # ========================================================

    for name, model in models.items():

        print("\n" + "=" * 75)
        print("GRID SEARCH:", name)
        print("=" * 75)

        # Avoid nested parallelism
        if name == "Random Forest":
            model.set_params(n_jobs=1)

        grid_search = GridSearchCV(

            estimator=model,

            param_grid=PARAM_GRIDS[name],

            scoring="roc_auc",

            cv=cv,

            n_jobs=-1,

            verbose=1,

            refit=True,

            return_train_score=True
        )

        # Fit only on training data
        grid_search.fit(
            X_train,
            y_train
        )

        best_model = grid_search.best_estimator_

        tuned_models[name] = best_model

        print("\nBEST HYPERPARAMETERS")

        print(grid_search.best_params_)

        print(
            "\nBest Cross-Validation ROC-AUC:",
            round(grid_search.best_score_, 4)
        )

        # ----------------------------------------------------
        # VALIDATION EVALUATION
        # ----------------------------------------------------

        results = evaluate_model(
            best_model,
            X_val,
            y_val
        )

        print("\nVALIDATION RESULTS")

        for metric, value in results.items():

            if metric != "Confusion Matrix":

                print(f"{metric}: {value:.4f}")

        print("\nConfusion Matrix:")

        print(results["Confusion Matrix"])

        validation_results.append({

            "Model": name,

            "Best CV ROC-AUC": grid_search.best_score_,

            "Validation Accuracy": results["Accuracy"],

            "Validation Precision": results["Precision"],

            "Validation Recall": results["Recall"],

            "Validation F1": results["F1"],

            "Validation ROC-AUC": results["ROC-AUC"],

            "Best Parameters": str(grid_search.best_params_)
        })

        # Save individual tuned model
        individual_model_path = os.path.join(
            MODEL_DIR,
            name.lower().replace(" ", "_")
            + "_gridsearch.pkl"
        )

        joblib.dump(
            best_model,
            individual_model_path
        )

        print("\nModel saved:", individual_model_path)

    # ========================================================
    # 7. MODEL COMPARISON
    # ========================================================

    comparison_df = pd.DataFrame(
        validation_results
    )

    comparison_path = os.path.join(
        RESULTS_DIR,
        "gridsearch_model_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False
    )

    print("\n" + "=" * 75)
    print("GRID SEARCH MODEL COMPARISON")
    print("=" * 75)

    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nComparison saved:", comparison_path)

    # ========================================================
    # 8. SELECT BEST MODEL
    # ========================================================

    best_index = comparison_df[
        "Validation ROC-AUC"
    ].idxmax()

    best_model_name = comparison_df.loc[
        best_index,
        "Model"
    ]

    best_model = tuned_models[best_model_name]

    print("\n" + "=" * 75)
    print("SELECTED MODEL")
    print("=" * 75)

    print("Best classifier:", best_model_name)

    # ========================================================
    # 9. RETRAIN USING TRAINING + VALIDATION DATA
    # ========================================================

    X_train_full = pd.concat(
        [X_train, X_val],
        axis=0
    )

    y_train_full = pd.concat(
        [y_train, y_val],
        axis=0
    )

    print("\nRetraining selected model...")

    final_model = clone(best_model)

    final_model.fit(
        X_train_full,
        y_train_full
    )

    print("Retraining completed.")

    # ========================================================
    # 10. FINAL TEST EVALUATION
    # ========================================================

    test_results = evaluate_model(
        final_model,
        X_test,
        y_test
    )

    print("\n" + "=" * 75)
    print("FINAL TEST RESULTS")
    print("=" * 75)

    for metric, value in test_results.items():

        if metric != "Confusion Matrix":

            print(f"{metric}: {value:.4f}")

    print("\nFinal Confusion Matrix:")

    print(test_results["Confusion Matrix"])

    # Save test metrics
    test_metrics_df = pd.DataFrame([

        {
            "Model": best_model_name,

            "Accuracy": test_results["Accuracy"],

            "Precision": test_results["Precision"],

            "Recall": test_results["Recall"],

            "F1": test_results["F1"],

            "ROC-AUC": test_results["ROC-AUC"]
        }

    ])

    test_metrics_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            "gridsearch_test_results.csv"
        ),
        index=False
    )

    # ========================================================
    # 11. GRAPH 1 - MODEL PERFORMANCE COMPARISON
    # ========================================================

    print("\nGenerating Graph 1...")

    metrics = [

        "Validation Accuracy",

        "Validation Precision",

        "Validation Recall",

        "Validation F1"
    ]

    comparison_df.set_index("Model")[metrics].plot(

        kind="bar",

        figsize=(12, 7),

        ylim=(0, 1),

        width=0.8
    )

    plt.title(
        "Cancer Screening - Model Performance Comparison"
    )

    plt.xlabel("Classifier")

    plt.ylabel("Score")

    plt.xticks(rotation=0)

    plt.legend(loc="lower right")

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "model_performance.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # 12. GRAPH 2 - ROC-AUC COMPARISON
    # ========================================================

    print("Generating Graph 2...")

    plt.figure(figsize=(10, 6))

    plt.bar(
        comparison_df["Model"],
        comparison_df["Validation ROC-AUC"]
    )

    plt.ylim(0, 1)

    plt.title(
        "ROC-AUC Comparison After GridSearchCV"
    )

    plt.xlabel("Classifier")

    plt.ylabel("ROC-AUC Score")

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "roc_auc_comparison.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # 13. GRAPH 3 - ROC CURVES
    # ========================================================

    print("Generating Graph 3...")

    plt.figure(figsize=(10, 7))

    ax = plt.gca()

    for name, model in tuned_models.items():

        RocCurveDisplay.from_estimator(

            model,

            X_val,

            y_val,

            name=name,

            ax=ax
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.title(
        "ROC Curves - Tuned Classification Models"
    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "roc_curves.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # 14. GRAPH 4 - CONFUSION MATRIX
    # ========================================================

    print("Generating Graph 4...")

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ConfusionMatrixDisplay.from_predictions(

        y_test,

        final_model.predict(X_test),

        display_labels=[
            "No Cancer",
            "Cancer"
        ],

        cmap="Blues",

        ax=ax
    )

    plt.title(
        f"Confusion Matrix - {best_model_name}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "confusion_matrix.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # 15. SAVE FINAL MODEL
    # ========================================================

    final_model_path = os.path.join(

        MODEL_DIR,

        "best_cancer_classifier_gridsearch.pkl"
    )

    joblib.dump(
        final_model,
        final_model_path
    )

    print("\nFinal model saved:", final_model_path)

    # ========================================================
    # 16. COMPLETION
    # ========================================================

    print("\n" + "=" * 75)

    print("GRIDSEARCHCV COMPLETED SUCCESSFULLY")

    print("=" * 75)

    print("\nGenerated Graphs:")

    print("1. model_performance.png")

    print("2. roc_auc_comparison.png")

    print("3. roc_curves.png")

    print("4. confusion_matrix.png")

    print("\nGraph Directory:")

    print(PLOT_DIR)

    print("\nFinal Model:")

    print(final_model_path)