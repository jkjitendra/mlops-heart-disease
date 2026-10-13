"""Track previously selected tuned model configurations in MLflow."""

import argparse

from src.experiment_tracking import track_cv_run
from src.train import (
    build_logistic_model,
    build_random_forest_model,
)


def main():
    parser = argparse.ArgumentParser(
        description="Track a tuned model using MLflow"
    )

    parser.add_argument(
        "--model",
        choices=["logistic", "forest"],
        required=True,
    )

    args = parser.parse_args()

    if args.model == "logistic":
        model = build_logistic_model()

        model.set_params(
            classifier__C=0.1,
            classifier__class_weight="balanced",
        )

        model_name = "LogisticRegression"
        run_name = "logistic-regression-tuned"

    else:
        model = build_random_forest_model()

        model.set_params(
            classifier__max_depth=5,
            classifier__min_samples_leaf=3,
            classifier__class_weight="balanced",
        )

        model_name = "RandomForestClassifier"
        run_name = "random-forest-tuned"

    track_cv_run(
        model_name=model_name,
        model=model,
        run_name=run_name,
        stage="tuned",
    )


if __name__ == "__main__":
    main()
