"""Hyperparameter tuning for the Random Forest classifier."""

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.data import RANDOM_STATE, load_data, split_data
from src.train import build_random_forest_model, SCORING


def main():
    X, y = load_data()

    # Never use the held-out test set during tuning.
    X_train, _, y_train, _ = split_data(X, y)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    param_grid = {
        "classifier__max_depth": [None, 5],
        "classifier__min_samples_leaf": [1, 3],
        "classifier__class_weight": [None, "balanced"],
    }

    search = GridSearchCV(
        estimator=build_random_forest_model(),
        param_grid=param_grid,
        scoring=SCORING,
        refit="roc_auc",
        cv=cv,
        n_jobs=1,
        error_score="raise",
    )

    print("=== RANDOM FOREST HYPERPARAMETER TUNING ===")
    print(f"Training records: {len(X_train)}")
    print("Parameter combinations: 8")
    print("Cross-validation folds: 5")
    print("Number of trees: 200")
    print("Selection metric: ROC-AUC")
    print("Held-out test set: Not used")

    search.fit(X_train, y_train)

    results = search.cv_results_

    ranked_indices = sorted(
        range(len(results["params"])),
        key=lambda i: results["rank_test_roc_auc"][i],
    )

    print("\n=== GRID SEARCH RESULTS ===")

    for i in ranked_indices:
        params = results["params"][i]

        print(
            f"\nRank {results['rank_test_roc_auc'][i]}: "
            f"max_depth={params['classifier__max_depth']}, "
            f"min_samples_leaf="
            f"{params['classifier__min_samples_leaf']}, "
            f"class_weight="
            f"{params['classifier__class_weight']}"
        )

        for metric in SCORING:
            mean = results[f"mean_test_{metric}"][i]
            std = results[f"std_test_{metric}"][i]

            print(
                f"  {metric}: {mean:.4f} "
                f"(std: {std:.4f})"
            )

    print("\n=== BEST CONFIGURATION ===")
    print("Best parameters:", search.best_params_)
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")

    print("\nRandom Forest tuning completed!")


if __name__ == "__main__":
    main()
