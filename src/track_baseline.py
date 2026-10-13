"""Track Logistic Regression baseline using MLflow."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow

from mlflow.tracking import MlflowClient
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data import RANDOM_STATE, load_data, split_data
from src.train import SCORING, build_logistic_model


TRACKING_URI = "http://127.0.0.1:5050"
EXPERIMENT_NAME = "heart-disease-model-comparison"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_data()
    X_train, _, y_train, _ = split_data(X, y)

    model = build_logistic_model()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    print("=== MLFLOW LOGISTIC REGRESSION BASELINE ===")
    print(f"Training records: {len(X_train)}")
    print("CV folds: 5")
    print("Held-out test set: Not used")

    with mlflow.start_run(
        run_name="logistic-regression-baseline"
    ) as run:

        classifier = model.named_steps["classifier"]

        # Record the model configuration.
        mlflow.log_params({
            "model": "LogisticRegression",
            "C": classifier.C,
            "class_weight": str(classifier.class_weight),
            "max_iter": classifier.max_iter,
            "random_state": RANDOM_STATE,
            "cv_folds": 5,
            "training_records": len(X_train),
        })

        mlflow.set_tags({
            "dataset": "UCI Cleveland Heart Disease",
            "evaluation": "training_cross_validation",
            "model_stage": "baseline",
        })

        # Evaluate using the training subset only.
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

        print("\n=== CROSS-VALIDATION METRICS ===")

        for metric in SCORING:
            scores = results[f"test_{metric}"]

            mean = float(scores.mean())
            std = float(scores.std())

            fold_scores[metric] = scores.tolist()

            mlflow.log_metric(f"cv_{metric}_mean", mean)
            mlflow.log_metric(f"cv_{metric}_std", std)

            print(f"{metric}: {mean:.4f} (std: {std:.4f})")

        # Save the individual fold scores as an artifact.
        mlflow.log_dict(
            fold_scores,
            "cross_validation/fold_scores.json",
        )

        # Create a plot showing metric variation across folds.
        fig, ax = plt.subplots(figsize=(9, 5))

        for metric, scores in fold_scores.items():
            ax.plot(
                range(1, 6),
                scores,
                marker="o",
                label=metric,
            )

        ax.set_title(
            "Logistic Regression - Cross-Validation Metrics"
        )
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

    # Verify the run was saved in MLflow.
    client = MlflowClient(tracking_uri=TRACKING_URI)

    saved_run = client.get_run(run_id)
    artifacts = client.list_artifacts(run_id)

    assert saved_run.data.metrics["cv_roc_auc_mean"] > 0

    print("\n=== MLFLOW RUN VERIFICATION ===")
    print("Run ID:", run_id)
    print("Run status:", saved_run.info.status)
    print("Artifacts:", [item.path for item in artifacts])

    print("\nMLflow baseline tracking completed!")


if __name__ == "__main__":
    main()
