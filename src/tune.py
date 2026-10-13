"""Hyperparameter tuning for Logistic Regression."""

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.data import RANDOM_STATE, load_data, split_data
from src.train import build_logistic_model, SCORING


def main():
    X, y = load_data()

    # Only training data is used for hyperparameter tuning.
    X_train, _, y_train, _ = split_data(X, y)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    param_grid = {
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__class_weight": [None, "balanced"],
    }

    search = GridSearchCV(
        estimator=build_logistic_model(),
        param_grid=param_grid,
        scoring=SCORING,
        refit="roc_auc",
        cv=cv,
        n_jobs=1,
        error_score="raise",
    )

    print("=== LOGISTIC REGRESSION TUNING ===")
    print(f"Training records: {len(X_train)}")
    print("Parameter combinations: 6")
    print("Cross-validation folds: 5")
    print("Selection metric: ROC-AUC")
    print("Held-out test set: Not used")

    search.fit(X_train, y_train)

    print("\n=== GRID SEARCH RESULTS ===")

    results = search.cv_results_

    ranked_indices = sorted(
        range(len(results["params"])),
        key=lambda i: results["mean_test_roc_auc"][i],
        reverse=True,
    )

    for rank, i in enumerate(ranked_indices, start=1):
        params = results["params"][i]

        print(
            f"\nRank {rank}: "
            f"C={params['classifier__C']}, "
            f"class_weight={params['classifier__class_weight']}"
        )

        for metric in SCORING:
            print(
                f"  {metric}: "
                f"{results[f'mean_test_{metric}'][i]:.4f}"
            )

    print("\n=== BEST CONFIGURATION ===")
    print("Best parameters:", search.best_params_)
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")

    print("\nLogistic Regression tuning completed!")


if __name__ == "__main__":
    main()
