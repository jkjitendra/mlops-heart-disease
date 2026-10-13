"""Reusable MLflow tracking for classification experiments."""

import hashlib

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data import DATA_PATH, RANDOM_STATE, load_data, split_data
from src.train import SCORING, build_random_forest_model


TRACKING_URI = "http://127.0.0.1:5050"
EXPERIMENT_NAME = "heart-disease-model-comparison"


def track_cv_run(model_name, model, run_name, stage="baseline"):
    """Record a model's training-only cross-validation results."""

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_data()
    X_train, _, y_train, _ = split_data(X, y)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    classifier_params = model.named_steps[
        "classifier"
    ].get_params(deep=False)

    print(f"\n=== TRACKING {model_name.upper()} ===")
    print(f"Training records: {len(X_train)}")
    print("Cross-validation folds: 5")
    print("Held-out test set: Not used")

    with mlflow.start_run(run_name=run_name) as run:

        mlflow.log_params({
            "model": model_name,
            "training_records": len(X_train),
            "cv_folds": 5,
            "random_state": RANDOM_STATE,
            "processed_data_sha256": hashlib.sha256(
                DATA_PATH.read_bytes()
            ).hexdigest(),
        })

        mlflow.log_params({
            f"classifier__{key}": str(value)
            for key, value in classifier_params.items()
        })

        mlflow.set_tags({
            "dataset": "UCI Cleveland Heart Disease",
            "evaluation": "training_cross_validation",
            "model_stage": stage,
        })

        results = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=SCORING,
            n_jobs=1,
            error_score="raise",
        )

        fold_scores = {}

        print("\n=== CROSS-VALIDATION RESULTS ===")

        fig, ax = plt.subplots(figsize=(9, 5))

        for metric in SCORING:
            scores = results[f"test_{metric}"]

            mean = float(scores.mean())
            std = float(scores.std())

            fold_scores[metric] = scores.tolist()

            mlflow.log_metric(f"cv_{metric}_mean", mean)
            mlflow.log_metric(f"cv_{metric}_std", std)

            print(f"{metric}: {mean:.4f} (std: {std:.4f})")

            ax.plot(
                range(1, 6),
                scores,
                marker="o",
                label=metric,
            )

        mlflow.log_dict(
            fold_scores,
            "cross_validation/fold_scores.json",
        )

        ax.set_title(f"{model_name} - CV Performance")
        ax.set_xlabel("Cross-Validation Fold")
        ax.set_ylabel("Score")
        ax.set_xticks(range(1, 6))
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend()

        mlflow.log_figure(
            fig,
            "plots/cv_metric_folds.png",
        )

        plt.close(fig)
        run_id = run.info.run_id

    client = MlflowClient(tracking_uri=TRACKING_URI)
    saved_run = client.get_run(run_id)

    artifact_paths = {
        item.path
        for item in client.list_artifacts(run_id)
    }

    assert saved_run.info.status == "FINISHED"
    assert {"cross_validation", "plots"}.issubset(
        artifact_paths
    )

    print("\n=== MLFLOW VERIFICATION ===")
    print("Run ID:", run_id)
    print("Run status:", saved_run.info.status)
    print("Artifacts:", sorted(artifact_paths))
    print("Experiment tracking completed!")

    return saved_run


if __name__ == "__main__":
    track_cv_run(
        model_name="RandomForestClassifier",
        model=build_random_forest_model(),
        run_name="random-forest-baseline",
    )
