"""Train and package the selected Random Forest in MLflow."""

import hashlib

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from src.data import (
    DATA_PATH,
    RANDOM_STATE,
    load_data,
    split_data,
)
from src.experiment_tracking import TRACKING_URI
from src.train import build_random_forest_model


EXPERIMENT_NAME = "heart-disease-model-packaging"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load the reproducible dataset split.
    X, y = load_data()
    X_train, _, y_train, _ = split_data(X, y)

    # Build the previously selected Random Forest.
    model = build_random_forest_model()

    model.set_params(
        classifier__max_depth=5,
        classifier__min_samples_leaf=3,
        classifier__class_weight="balanced",
    )

    print("=== MODEL PACKAGING ===")
    print("Model: Tuned Random Forest")
    print(f"Training records: {len(X_train)}")
    print("Held-out test set: Not used")

    # Fit preprocessing and classifier on training data.
    model.fit(X_train, y_train)

    # Prepare model signature and sample input.
    input_example = X_train.head(5).copy()
    example_predictions = model.predict(input_example)

    signature = infer_signature(
        input_example,
        example_predictions,
    )

    with mlflow.start_run(
        run_name="tuned-random-forest-packaged"
    ) as run:

        # Record model configuration and dataset identity.
        mlflow.log_params({
            "model": "RandomForestClassifier",
            "n_estimators": 200,
            "max_depth": 5,
            "min_samples_leaf": 3,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "training_records": len(X_train),
            "selection_metric": "cv_roc_auc",
            "processed_data_sha256": hashlib.sha256(
                DATA_PATH.read_bytes()
            ).hexdigest(),
        })

        mlflow.set_tags({
            "dataset": "UCI Cleveland Heart Disease",
            "model_stage": "packaging",
            "evaluation": "previously_selected_configuration",
            "intended_use": "academic_demonstration",
            "selected_cv_run_id":
                "3bf028ad8b7b45e89766e325703d38f3",
        })

        # Save the entire fitted pipeline.
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="heart-disease-random-forest",
            signature=signature,
            input_example=input_example,
            serialization_format="skops",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree",
            ],
        )

        run_id = run.info.run_id

    print("\n=== PACKAGED MODEL ===")
    print("MLflow run ID:", run_id)
    print("MLflow model ID:", model_info.model_id)
    print("Model URI:", model_info.model_uri)
    print("Input features:", input_example.shape[1])

    print("\nModel packaging completed successfully!")


if __name__ == "__main__":
    main()
