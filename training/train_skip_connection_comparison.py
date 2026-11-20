import os
import copy
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
BATCH_SIZE = 32
HIDDEN_DIM = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 15
THRESHOLD = 0.50

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
TARGET_COLUMN = "Diagnosis"

# Project root is the parent of the directory containing this script.
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT, "results", "cancer_screening"
)
MODEL_DIR = os.path.join(
    PROJECT_ROOT, "saved_models", "cancer_screening"
)
OUTPUT_DIR = os.path.join(
    RESULTS_DIR, "skip_connection_experiment"
)
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic settings improve reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 3. LOAD THE EXISTING TRAIN / VALIDATION / TEST SPLITS
# ============================================================

def load_data():
    paths = {
        "train": os.path.join(RESULTS_DIR, "train_processed.csv"),
        "validation": os.path.join(RESULTS_DIR, "validation_processed.csv"),
        "test": os.path.join(RESULTS_DIR, "test_processed.csv"),
    }

    for split, path in paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing {split} CSV: {path}\n"
                "Check PROJECT_ROOT and the results/cancer_screening folder."
            )

    train_df = pd.read_csv(paths["train"])
    val_df = pd.read_csv(paths["validation"])
    test_df = pd.read_csv(paths["test"])

    expected = FEATURE_COLUMNS + [TARGET_COLUMN]

    for split_name, frame in [
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ]:
        missing = [col for col in expected if col not in frame.columns]
        if missing:
            raise ValueError(
                f"{split_name} CSV is missing columns: {missing}. "
                f"Found columns: {frame.columns.tolist()}"
            )

        if frame[expected].isna().any().any():
            raise ValueError(f"Missing values found in {split_name} data.")

    X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

    X_val = val_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_val = val_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

    X_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

    for split_name, labels in [
        ("train", y_train),
        ("validation", y_val),
        ("test", y_test),
    ]:
        unique_labels = set(np.unique(labels).tolist())
        if not unique_labels.issubset({0.0, 1.0}):
            raise ValueError(
                f"{split_name} target must contain binary labels 0 and 1; "
                f"found {unique_labels}"
            )

    return (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
    )


# ============================================================
# 4. MODEL A: WITHOUT SKIP CONNECTION
# ============================================================

class MLPWithoutSkip(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


# ============================================================
# 5. MODEL B: WITH RESIDUAL SKIP CONNECTION
# ============================================================

class MLPWithSkip(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)

        # Projection creates a matching hidden_dim representation
        # so it can be added to the second hidden layer.
        self.skip_projection = nn.Linear(hidden_dim, hidden_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.20)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h1 = self.relu(self.input_layer(x))
        h1 = self.dropout(h1)

        main_path = self.hidden_layer(h1)
        skip_path = self.skip_projection(h1)

        # Residual / skip addition
        h2 = self.relu(main_path + skip_path)
        h2 = self.dropout(h2)

        return self.output_layer(h2).squeeze(1)


# ============================================================
# 6. DATA LOADERS
# ============================================================

def make_loader(X, y, shuffle):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )


# ============================================================
# 7. EVALUATION HELPERS
# ============================================================

def collect_predictions(model, loader, criterion=None):
    model.eval()

    all_labels = []
    all_probabilities = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(features)

            if criterion is not None:
                loss = criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)

            probabilities = torch.sigmoid(logits)

            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            total_samples += labels.size(0)

    labels = np.asarray(all_labels).astype(int)
    probabilities = np.asarray(all_probabilities)
    predictions = (probabilities >= THRESHOLD).astype(int)

    average_loss = (
        total_loss / total_samples
        if criterion is not None and total_samples > 0
        else None
    )

    return labels, probabilities, predictions, average_loss


def calculate_metrics(labels, probabilities, predictions):
    return {
        "Accuracy": accuracy_score(labels, predictions),
        "Precision": precision_score(
            labels, predictions, zero_division=0
        ),
        "Recall": recall_score(
            labels, predictions, zero_division=0
        ),
        "F1": f1_score(
            labels, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(labels, probabilities),
    }


# ============================================================
# 8. TRAIN ONE MODEL WITH EARLY STOPPING ON VALIDATION LOSS
# ============================================================

def train_model(model_name, model, train_loader, val_loader, y_train):
    print("\n" + "=" * 72)
    print(f"TRAINING: {model_name}")
    print("=" * 72)

    model = model.to(DEVICE)

    # Balanced class weights are derived from training labels only.
    classes = np.array([0, 1])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train.astype(int),
    )
    class_weights = torch.tensor(
        weights, dtype=torch.float32, device=DEVICE
    )

    # Per-sample weights: negative class weight for label 0,
    # positive class weight for label 1.
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=class_weights[1] / class_weights[0]
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        running_loss = 0.0
        train_labels = []
        train_probabilities = []

        for features, labels in train_loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            logits = model(features)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss in {model_name}, epoch {epoch}."
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=5.0
            )
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

            train_labels.extend(labels.detach().cpu().numpy())
            train_probabilities.extend(
                torch.sigmoid(logits).detach().cpu().numpy()
            )

        train_loss = running_loss / len(train_loader.dataset)

        train_labels_np = np.asarray(train_labels).astype(int)
        train_predictions = (
            np.asarray(train_probabilities) >= THRESHOLD
        ).astype(int)

        train_accuracy = accuracy_score(
            train_labels_np, train_predictions
        )

        # Validation
        val_labels, val_probabilities, val_predictions, val_loss = (
            collect_predictions(model, val_loader, criterion)
        )

        val_accuracy = accuracy_score(
            val_labels, val_predictions
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch [{epoch:03d}/{MAX_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        # Save the best weights according to validation loss.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(
                f"Early stopping: validation loss did not improve for "
                f"{PATIENCE} consecutive epochs."
            )
            break

    if best_state is None:
        raise RuntimeError(f"No best checkpoint was recorded for {model_name}.")

    model.load_state_dict(best_state)

    safe_name = model_name.lower().replace(" ", "_")
    checkpoint_path = os.path.join(
        MODEL_DIR, f"{safe_name}_skip_experiment.pth"
    )

    torch.save(
        {
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "input_dim": len(FEATURE_COLUMNS),
            "hidden_dim": HIDDEN_DIM,
            "feature_columns": FEATURE_COLUMNS,
            "best_epoch": best_epoch,
            "best_validation_loss": best_val_loss,
            "threshold": THRESHOLD,
        },
        checkpoint_path,
    )

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved: {checkpoint_path}")

    return model, history, best_epoch


# ============================================================
# 9. PLOT TRAINING CURVES
# ============================================================

def plot_training_curves(histories):
    # Loss curves
    plt.figure(figsize=(10, 6))

    for name, history in histories.items():
        epochs = range(1, len(history["train_loss"]) + 1)
        plt.plot(
            epochs, history["train_loss"],
            label=f"{name} - Train"
        )
        plt.plot(
            epochs, history["val_loss"],
            linestyle="--",
            label=f"{name} - Validation"
        )

    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, "training_validation_loss.png"),
        dpi=300
    )
    plt.close()

    # Accuracy curves
    plt.figure(figsize=(10, 6))

    for name, history in histories.items():
        epochs = range(1, len(history["train_accuracy"]) + 1)
        plt.plot(
            epochs, history["train_accuracy"],
            label=f"{name} - Train"
        )
        plt.plot(
            epochs, history["val_accuracy"],
            linestyle="--",
            label=f"{name} - Validation"
        )

    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, "training_validation_accuracy.png"),
        dpi=300
    )
    plt.close()


# ============================================================
# 10. PLOT TEST ROC CURVES
# ============================================================

def plot_roc_curves(test_data):
    plt.figure(figsize=(9, 7))
    ax = plt.gca()

    for name, data in test_data.items():
        RocCurveDisplay.from_predictions(
            data["labels"],
            data["probabilities"],
            name=name,
            ax=ax,
        )

    ax.plot(
        [0, 1], [0, 1],
        linestyle="--",
        label="Random classifier"
    )

    plt.title("Test ROC Curves: With vs Without Skip Connection")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, "test_roc_curves.png"),
        dpi=300
    )
    plt.close()


# ============================================================
# 11. PLOT CONFUSION MATRICES
# ============================================================

def plot_confusion_matrices(test_data):
    for name, data in test_data.items():
        cm = confusion_matrix(
            data["labels"],
            data["predictions"],
            labels=[0, 1],
        )

        fig, ax = plt.subplots(figsize=(7, 6))

        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No Cancer", "Cancer"],
        ).plot(
            cmap="Blues",
            values_format="d",
            ax=ax,
            colorbar=False,
        )

        ax.set_title(f"Test Confusion Matrix - {name}")
        plt.tight_layout()

        filename = (
            name.lower().replace(" ", "_")
            .replace("-", "_")
            + "_confusion_matrix.png"
        )

        plt.savefig(
            os.path.join(PLOT_DIR, filename),
            dpi=300
        )
        plt.close()


# ============================================================
# 12. PLOT METRIC COMPARISON
# ============================================================

def plot_metric_comparison(results_df):
    metric_columns = [
        "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"
    ]

    results_df.set_index("Model")[metric_columns].plot(
        kind="bar",
        figsize=(11, 7),
        ylim=(0, 1.05),
        width=0.75,
    )

    plt.title("Test Performance: With vs Without Skip Connection")
    plt.xlabel("Architecture")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, "test_metric_comparison.png"),
        dpi=300
    )
    plt.close()


# ============================================================
# 13. MAIN
# ============================================================

def main():
    set_seed(SEED)

    print("=" * 72)
    print("SKIP CONNECTION ABLATION EXPERIMENT")
    print("=" * 72)
    print("Device:", DEVICE)
    print("Project root:", PROJECT_ROOT)

    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
    ) = load_data()

    print("\nDataset sizes:")
    print("Train:", len(X_train))
    print("Validation:", len(X_val))
    print("Test:", len(X_test))
    print("Input features:", len(FEATURE_COLUMNS))
    print("Feature order:", FEATURE_COLUMNS)

    # The preprocessed splits are used as supplied.
    train_loader = make_loader(X_train, y_train, shuffle=True)
    train_eval_loader = make_loader(X_train, y_train, shuffle=False)
    val_loader = make_loader(X_val, y_val, shuffle=False)
    test_loader = make_loader(X_test, y_test, shuffle=False)

    # Same seed before each initialization makes the comparison
    # more reproducible. Model structures still differ by the skip branch.
    set_seed(SEED)
    model_without = MLPWithoutSkip(
        input_dim=X_train.shape[1]
    )

    model_without, history_without, best_epoch_without = train_model(
        "Without Skip Connection",
        model_without,
        train_loader,
        val_loader,
        y_train,
    )

    # Re-create shuffled loader with deterministic seed for second run.
    set_seed(SEED)
    train_loader_second = make_loader(
        X_train, y_train, shuffle=True
    )

    model_with = MLPWithSkip(
        input_dim=X_train.shape[1]
    )

    model_with, history_with, best_epoch_with = train_model(
        "With Skip Connection",
        model_with,
        train_loader_second,
        val_loader,
        y_train,
    )

    histories = {
        "Without Skip": history_without,
        "With Skip": history_with,
    }

    # --------------------------------------------------------
    # Validation results (for analysis; not used to tune test)
    # --------------------------------------------------------
    validation_results = []

    for name, model in [
        ("Without Skip Connection", model_without),
        ("With Skip Connection", model_with),
    ]:
        labels, probabilities, predictions, _ = collect_predictions(
            model, val_loader
        )
        metrics = calculate_metrics(
            labels, probabilities, predictions
        )
        validation_results.append({
            "Model": name,
            **metrics,
        })

    validation_df = pd.DataFrame(validation_results)
    validation_df.to_csv(
        os.path.join(OUTPUT_DIR, "validation_comparison.csv"),
        index=False,
    )

    # --------------------------------------------------------
    # Final test evaluation: evaluate each selected checkpoint
    # once on the held-out test split.
    # --------------------------------------------------------
    test_data = {}
    test_results = []

    for name, model in [
        ("Without Skip Connection", model_without),
        ("With Skip Connection", model_with),
    ]:
        labels, probabilities, predictions, _ = collect_predictions(
            model, test_loader
        )

        metrics = calculate_metrics(
            labels, probabilities, predictions
        )

        test_data[name] = {
            "labels": labels,
            "probabilities": probabilities,
            "predictions": predictions,
        }

        test_results.append({
            "Model": name,
            **metrics,
        })

    results_df = pd.DataFrame(test_results)

    results_df.to_csv(
        os.path.join(OUTPUT_DIR, "test_comparison.csv"),
        index=False,
    )

    # Save histories for reproducible reporting.
    history_rows = []
    for model_name, history in histories.items():
        for epoch_idx in range(len(history["train_loss"])):
            history_rows.append({
                "Model": model_name,
                "Epoch": epoch_idx + 1,
                "Train Loss": history["train_loss"][epoch_idx],
                "Validation Loss": history["val_loss"][epoch_idx],
                "Train Accuracy": history["train_accuracy"][epoch_idx],
                "Validation Accuracy": history["val_accuracy"][epoch_idx],
            })

    pd.DataFrame(history_rows).to_csv(
        os.path.join(OUTPUT_DIR, "training_history.csv"),
        index=False,
    )

    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------
    print("\n" + "=" * 72)
    print("VALIDATION COMPARISON")
    print("=" * 72)
    print(validation_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 72)
    print("FINAL TEST COMPARISON")
    print("=" * 72)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nBest epochs:")
    print("Without skip:", best_epoch_without)
    print("With skip:", best_epoch_with)

    # --------------------------------------------------------
    # Graphs
    # --------------------------------------------------------
    print("\nGenerating graphs...")

    plot_training_curves(histories)
    plot_roc_curves(test_data)
    plot_confusion_matrices(test_data)
    plot_metric_comparison(results_df)

    # Save experiment metadata.
    metadata = {
        "seed": SEED,
        "device": str(DEVICE),
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "batch_size": BATCH_SIZE,
        "hidden_dim": HIDDEN_DIM,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "decision_threshold": THRESHOLD,
        "best_epoch_without_skip": best_epoch_without,
        "best_epoch_with_skip": best_epoch_with,
        "note": (
            "Uses the existing processed train/validation/test CSV splits. "
            "Test set was not used for model selection."
        ),
    }

    with open(
        os.path.join(OUTPUT_DIR, "experiment_metadata.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\n" + "=" * 72)
    print("EXPERIMENT COMPLETED")
    print("=" * 72)
    print("Results directory:", OUTPUT_DIR)
    print("Plots directory:", PLOT_DIR)
    print("Checkpoints saved in:", MODEL_DIR)


if __name__ == "__main__":
    main()
