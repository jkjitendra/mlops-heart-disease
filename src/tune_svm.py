"""Tune SVM hyperparameters and record results in MLflow."""

import hashlib

import mlflow
from mlflow.tracking import MlflowClient
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.data import (
    DATA_PATH,
    RANDOM_STATE,
    load_data,
    split_data,
)
from src.extra_models import build_svm_model
from src.experiment_tracking import (
    EXPERIMENT_NAME,
    TRACKING_URI,
)
from src.train import SCORING


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_data()

    # Only the training records are used.
    X_train, _, y_train, _ = split_data(X, y)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    param_grid = {
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__gamma": ["scale", 0.1],
        "classifier__class_weight": [None, "balanced"],
    }

    search = GridSearchCV(
        estimator=build_svm_model(),
        param_grid=param_grid,
        scoring=SCORING,
        refit="roc_auc",
        cv=cv,
        n_jobs=1,
        error_score="raise",
    )

    print("=== SVM HYPERPARAMETER TUNING ===")
    print(f"Training records: {len(X_train)}")
    print("Parameter combinations: 12")
    print("Cross-validation folds: 5")
    print("Selection metric: ROC-AUC")
    print("Held-out test set: Not used")

    with mlflow.start_run(run_name="svm-rbf-tuning") as run:

        mlflow.log_params({
            "model": "SVC",
            "kernel": "rbf",
            "training_records": len(X_train),
            "cv_folds": 5,
            "random_state": RANDOM_STATE,
            "selection_metric": "roc_auc",
            "parameter_combinations": 12,
            "processed_data_sha256": hashlib.sha256(
                DATA_PATH.read_bytes()
            ).hexdigest(),
        })

        mlflow.set_tags({
            "dataset": "UCI Cleveland Heart Disease",
            "model_stage": "hyperparameter_tuning",
            "evaluation": "training_cross_validation",
        })

        # Search and select using training records only.
        search.fit(X_train, y_train)

        results = search.cv_results_

        candidate_results = []

        for i, params in enumerate(results["params"]):
            entry = {
                "rank": int(
                    results["rank_test_roc_auc"][i]
                ),
                "parameters": params,
                "metrics": {},
            }

            for metric in SCORING:
                entry["metrics"][metric] = {
                    "mean": float(
                        results[f"mean_test_{metric}"][i]
                    ),
                    "std": float(
                        results[f"std_test_{metric}"][i]
                    ),
                }

            candidate_results.append(entry)

        candidate_results.sort(
            key=lambda item: item["rank"]
        )

        # Save all configurations, not just the winner.
        mlflow.log_dict(
            candidate_results,
            "tuning/all_candidate_results.json",
        )

        mlflow.log_dict(
            search.best_params_,
            "tuning/best_parameters.json",
        )

        for key, value in search.best_params_.items():
            mlflow.log_param(
                f"best_{key}",
                str(value),
            )

        best_index = search.best_index_

        print("\n=== BEST SVM CONFIGURATION ===")

        print("Best parameters:", search.best_params_)

        for metric in SCORING:
            mean = float(
                results[f"mean_test_{metric}"][best_index]
            )
            std = float(
                results[f"std_test_{metric}"][best_index]
            )

            mlflow.log_metric(f"cv_{metric}_mean", mean)
            mlflow.log_metric(f"cv_{metric}_std", std)

            print(
                f"{metric}: {mean:.4f} "
                f"(std: {std:.4f})"
            )

        print(
            f"\nBest CV ROC-AUC: {search.best_score_:.4f}"
        )

        run_id = run.info.run_id

    # Verify the completed MLflow run.
    client = MlflowClient(tracking_uri=TRACKING_URI)
    saved_run = client.get_run(run_id)
    artifacts = client.list_artifacts(run_id)

    assert saved_run.info.status == "FINISHED"
    assert any(item.path == "tuning" for item in artifacts)

    print("\n=== MLFLOW VERIFICATION ===")
    print("Run ID:", run_id)
    print("Run status:", saved_run.info.status)
    print("Artifacts:", [item.path for item in artifacts])
    print("\nSVM tuning completed successfully!")


if __name__ == "__main__":
    main()
