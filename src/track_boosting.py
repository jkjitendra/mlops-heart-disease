"""Track Gradient Boosting baseline using MLflow."""

from src.experiment_tracking import track_cv_run
from src.extra_models import build_gradient_boosting_model


def main():
    track_cv_run(
        model_name="GradientBoostingClassifier",
        model=build_gradient_boosting_model(),
        run_name="gradient-boosting-baseline",
        stage="baseline",
    )


if __name__ == "__main__":
    main()
