"""Train and compare heart disease classification models."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.data import RANDOM_STATE, load_data, split_data
from src.preprocessing import build_preprocessor


SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def build_logistic_model():
    """Create an unfitted Logistic Regression pipeline."""

    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_random_forest_model():
    """Create an unfitted Random Forest pipeline."""

    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )


def evaluate_model(name, model, X_train, y_train, cv):
    """Evaluate a model using stratified cross-validation."""

    results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=SCORING,
        n_jobs=1,
        error_score="raise",
    )

    print(f"\n=== {name.upper()} ===")

    summary = {}

    for metric in SCORING:
        scores = results[f"test_{metric}"]

        mean = scores.mean()
        std = scores.std()

        summary[metric] = mean

        print(f"\n{metric.upper()}")
        print("Fold scores:", scores.round(4))
        print(f"Mean: {mean:.4f}")
        print(f"Std:  {std:.4f}")

    return summary


def main():
    X, y = load_data()

    X_train, _, y_train, _ = split_data(X, y)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    models = {
        "Logistic Regression": build_logistic_model(),
        "Random Forest": build_random_forest_model(),
    }

    print("=== MODEL COMPARISON ===")
    print(f"Training records: {len(X_train)}")
    print(f"Cross-validation folds: {cv.n_splits}")
    print("Held-out test set: Not used")

    summaries = {}

    for name, model in models.items():
        summaries[name] = evaluate_model(
            name,
            model,
            X_train,
            y_train,
            cv,
        )

    print("\n=== COMPARISON SUMMARY (CV MEAN) ===")

    print(
        f"{'Model':<22}"
        f"{'Accuracy':>10}"
        f"{'Precision':>11}"
        f"{'Recall':>9}"
        f"{'F1':>9}"
        f"{'ROC-AUC':>10}"
    )

    for name, metrics in summaries.items():
        print(
            f"{name:<22}"
            f"{metrics['accuracy']:>10.4f}"
            f"{metrics['precision']:>11.4f}"
            f"{metrics['recall']:>9.4f}"
            f"{metrics['f1']:>9.4f}"
            f"{metrics['roc_auc']:>10.4f}"
        )

    print("\nModel comparison completed!")


if __name__ == "__main__":
    main()
