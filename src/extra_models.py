"""Additional classification models for heart disease prediction."""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.data import RANDOM_STATE
from src.preprocessing import build_preprocessor


def build_svm_model():
    """Create an unfitted SVM classification pipeline."""

    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessor()),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_gradient_boosting_model():
    """Create an unfitted Gradient Boosting pipeline."""

    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessor()),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
