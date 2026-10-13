"""Export baseline and tuned model results from MLflow."""

import csv
from pathlib import Path

from mlflow.tracking import MlflowClient

from src.experiment_tracking import (
    TRACKING_URI,
    EXPERIMENT_NAME,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "reports/mlflow_model_comparison.csv"

EXPECTED_RUNS = [
    ("Logistic Regression", "Baseline", "logistic-regression-baseline"),
    ("Logistic Regression", "Tuned", "logistic-regression-tuned"),
    ("Random Forest", "Baseline", "random-forest-baseline"),
    ("Random Forest", "Tuned", "random-forest-tuned"),
    ("SVM (RBF)", "Baseline", "svm-rbf-baseline"),
    ("SVM (RBF)", "Tuned", "svm-rbf-tuning"),
    ("Gradient Boosting", "Baseline", "gradient-boosting-baseline"),
    ("Gradient Boosting", "Tuned", "gradient-boosting-tuning"),
]

METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
]


def main():
    client = MlflowClient(tracking_uri=TRACKING_URI)

    experiment = client.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment not found: {EXPERIMENT_NAME}"
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=500,
    )

    # Retain the latest active, successful run for each name.
    runs_by_name = {}

    expected_names = {
        run_name for _, _, run_name in EXPECTED_RUNS
    }

    for run in runs:
        name = run.data.tags.get("mlflow.runName")

        if name not in expected_names:
            continue

        if run.info.status != "FINISHED":
            continue

        if run.info.lifecycle_stage != "active":
            continue

        previous = runs_by_name.get(name)

        if (
            previous is None
            or run.info.start_time > previous.info.start_time
        ):
            runs_by_name[name] = run

    missing = expected_names - set(runs_by_name)

    if missing:
        raise RuntimeError(
            f"Missing MLflow runs: {sorted(missing)}"
        )

    rows = []

    for model_name, configuration, run_name in EXPECTED_RUNS:
        run = runs_by_name[run_name]

        row = {
            "model": model_name,
            "configuration": configuration,
            "run_name": run_name,
            "run_id": run.info.run_id,
        }

        for metric in METRICS:
            key = f"cv_{metric}_mean"

            if key not in run.data.metrics:
                raise RuntimeError(
                    f"Missing metric {key} in {run_name}"
                )

            row[metric] = run.data.metrics[key]

        rows.append(row)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "model",
        "configuration",
        "run_name",
        "run_id",
        *METRICS,
    ]

    with OUTPUT_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print("=== MLFLOW EXPERIMENT EXPORT ===")
    print("Experiment:", EXPERIMENT_NAME)
    print("Total model runs:", len(rows))

    print("\n=== MODEL COMPARISON ===")
    print(
        f"{'Model':<22} "
        f"{'Config':<10} "
        f"{'Accuracy':>9} "
        f"{'Recall':>8} "
        f"{'ROC-AUC':>9}"
    )

    for row in rows:
        print(
            f"{row['model']:<22} "
            f"{row['configuration']:<10} "
            f"{row['accuracy']:>9.4f} "
            f"{row['recall']:>8.4f} "
            f"{row['roc_auc']:>9.4f}"
        )

    print("\nExported:", OUTPUT_PATH.relative_to(ROOT))
    print("MLflow experiment export completed!")


if __name__ == "__main__":
    main()
