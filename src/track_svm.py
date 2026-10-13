"""Track the Support Vector Machine baseline using MLflow."""

from src.experiment_tracking import track_cv_run
from src.extra_models import build_svm_model


def main():
    track_cv_run(
        model_name="SVC",
        model=build_svm_model(),
        run_name="svm-rbf-baseline-corrected",
        stage="baseline",
    )


if __name__ == "__main__":
    main()
