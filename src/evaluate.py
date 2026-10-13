"""Final held-out evaluation of the selected Random Forest."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data import load_data, split_data
from src.train import build_random_forest_model


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"


def main():
    # Load the same reproducible train/test split.
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Use the configuration selected BEFORE test evaluation.
    model = build_random_forest_model()

    model.set_params(
        classifier__max_depth=5,
        classifier__min_samples_leaf=3,
        classifier__class_weight="balanced",
    )

    print("=== FINAL HELD-OUT MODEL EVALUATION ===")
    print("Selected model: Tuned Random Forest")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    # Fit the entire pipeline on training data only.
    model.fit(X_train, y_train)

    # Make predictions on previously unseen records.
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(y_test, predictions)
        ),
        "recall": float(
            recall_score(y_test, predictions)
        ),
        "f1": float(
            f1_score(y_test, predictions)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
    }

    print("\n=== FINAL TEST METRICS ===")

    for name, value in metrics.items():
        print(f"{name.upper()}: {value:.4f}")

    print("\n=== CLASSIFICATION REPORT ===")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "No Heart Disease",
                "Heart Disease Present",
            ],
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    print("\n=== CONFUSION MATRIX ===")
    print(cm)

    tn, fp, fn, tp = cm.ravel()

    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")

    # Save exact metrics for reporting and reproducibility.
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = REPORT_DIR / "test_metrics.json"

    with metrics_path.open("w") as file:
        json.dump(
            {
                "model": "Tuned Random Forest",
                "training_records": len(X_train),
                "testing_records": len(X_test),
                "metrics": metrics,
                "confusion_matrix": cm.tolist(),
            },
            file,
            indent=2,
        )

    # Confusion matrix visualization.
    fig, ax = plt.subplots(figsize=(7, 6))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Disease", "Disease"],
    )

    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Random Forest - Test Confusion Matrix")

    fig.savefig(
        FIGURE_DIR / "test_confusion_matrix.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ROC curve visualization.
    fig, ax = plt.subplots(figsize=(7, 6))

    RocCurveDisplay.from_predictions(
        y_test,
        probabilities,
        ax=ax,
        name="Tuned Random Forest",
    )

    ax.plot([0, 1], [0, 1], "--", label="Random Classifier")
    ax.set_title("Random Forest - Test ROC Curve")
    ax.legend()

    fig.savefig(
        FIGURE_DIR / "test_roc_curve.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\n=== SAVED EVALUATION ARTIFACTS ===")
    print("reports/test_metrics.json")
    print("reports/figures/test_confusion_matrix.png")
    print("reports/figures/test_roc_curve.png")

    print("\nFinal model evaluation completed!")


if __name__ == "__main__":
    main()
