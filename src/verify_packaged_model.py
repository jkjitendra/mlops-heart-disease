"""Verify the previously packaged MLflow Random Forest."""

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np

from src.data import load_data, split_data
from src.experiment_tracking import TRACKING_URI
from src.train import build_random_forest_model


MODEL_URI = (
    "models:/m-b853cd78dd7b4fbf8acfd01e414a817a"
)


def main():
    mlflow.set_tracking_uri(TRACKING_URI)

    print("=== LOADING PACKAGED MODEL ===")
    print("Model URI:", MODEL_URI)

    # Load the original fitted pipeline from MLflow.
    saved_model = mlflow.sklearn.load_model(MODEL_URI)

    # Use only training records for verification.
    X, y = load_data()
    X_train, _, y_train, _ = split_data(X, y)

    assert len(X_train) == 242
    assert X_train.shape[1] == 13

    # Verify the saved classifier configuration.
    params = saved_model.get_params()

    assert params["classifier__n_estimators"] == 200
    assert params["classifier__max_depth"] == 5
    assert params["classifier__min_samples_leaf"] == 3
    assert params["classifier__class_weight"] == "balanced"
    assert params["classifier__random_state"] == 42

    print("Saved hyperparameters: VERIFIED")

    # Train a fresh copy using exactly the same configuration.
    local_model = build_random_forest_model()

    local_model.set_params(
        classifier__max_depth=5,
        classifier__min_samples_leaf=3,
        classifier__class_weight="balanced",
    )

    local_model.fit(X_train, y_train)

    # Compare predictions on training samples only.
    sample = X_train.head(10).copy()

    original_predictions = local_model.predict(sample)
    loaded_predictions = saved_model.predict(sample)

    np.testing.assert_array_equal(
        original_predictions,
        loaded_predictions,
    )

    print("Native predictions: IDENTICAL")

    # Verify predicted probabilities as well.
    original_probabilities = local_model.predict_proba(
        sample
    )

    loaded_probabilities = saved_model.predict_proba(
        sample
    )

    np.testing.assert_allclose(
        original_probabilities,
        loaded_probabilities,
        rtol=1e-10,
        atol=1e-12,
    )

    print("Prediction probabilities: IDENTICAL")

    # Test MLflow's generic deployment interface.
    pyfunc_model = mlflow.pyfunc.load_model(MODEL_URI)

    pyfunc_predictions = pyfunc_model.predict(sample)

    np.testing.assert_array_equal(
        loaded_predictions,
        np.asarray(pyfunc_predictions),
    )

    print("MLflow PyFunc predictions: IDENTICAL")

    print("\n=== SAMPLE PREDICTIONS ===")
    print("Records checked:", len(sample))
    print("Predictions:", loaded_predictions.tolist())

    print("\n=== VERIFICATION SUMMARY ===")
    print("Training records: 242")
    print("Input features: 13")
    print("Held-out test set: Not used")
    print("Model load: PASS")
    print("Hyperparameter verification: PASS")
    print("Prediction consistency: PASS")
    print("Probability consistency: PASS")
    print("MLflow PyFunc compatibility: PASS")
    print("\nPackaged model verification completed!")


if __name__ == "__main__":
    main()
