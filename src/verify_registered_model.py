"""Verify MLflow registered model Version 1."""

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import numpy as np

from mlflow.tracking import MlflowClient

from src.data import load_data, split_data
from src.experiment_tracking import TRACKING_URI


REGISTERED_MODEL_NAME = "heart-disease-classifier"
MODEL_VERSION = "1"

SOURCE_MODEL_URI = (
    "models:/m-b853cd78dd7b4fbf8acfd01e414a817a"
)

REGISTERED_MODEL_URI = (
    f"models:/{REGISTERED_MODEL_NAME}/{MODEL_VERSION}"
)

EXPECTED_RUN_ID = "7fdb1783f308430da79ab15b5d939078"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)

    client = MlflowClient(tracking_uri=TRACKING_URI)

    print("=== REGISTERED MODEL VERIFICATION ===")
    print("Registered Model URI:", REGISTERED_MODEL_URI)

    # Check registry metadata before loading.
    version = client.get_model_version(
        REGISTERED_MODEL_NAME,
        MODEL_VERSION,
    )

    assert version.status == "READY"
    assert version.source == SOURCE_MODEL_URI
    assert version.run_id == EXPECTED_RUN_ID

    print("Registry metadata: VERIFIED")

    # Load original model and its registered version.
    source_model = mlflow.sklearn.load_model(
        SOURCE_MODEL_URI
    )

    registered_model = mlflow.sklearn.load_model(
        REGISTERED_MODEL_URI
    )

    print("Registered model loading: PASS")

    # Verify using training samples only.
    X, y = load_data()
    X_train, _, _, _ = split_data(X, y)

    sample = X_train.head(10).copy()

    # Compare class predictions.
    source_predictions = source_model.predict(sample)
    registered_predictions = registered_model.predict(
        sample
    )

    np.testing.assert_array_equal(
        source_predictions,
        registered_predictions,
    )

    print("Class predictions: IDENTICAL")

    # Compare predicted probabilities.
    source_probabilities = source_model.predict_proba(
        sample
    )

    registered_probabilities = (
        registered_model.predict_proba(sample)
    )

    np.testing.assert_allclose(
        source_probabilities,
        registered_probabilities,
        rtol=1e-10,
        atol=1e-12,
    )

    print("Prediction probabilities: IDENTICAL")

    # Verify MLflow's deployment-friendly interface.
    pyfunc_model = mlflow.pyfunc.load_model(
        REGISTERED_MODEL_URI
    )

    pyfunc_predictions = pyfunc_model.predict(sample)

    np.testing.assert_array_equal(
        registered_predictions,
        np.asarray(pyfunc_predictions),
    )

    print("MLflow PyFunc predictions: IDENTICAL")

    print("\n=== VERIFICATION SUMMARY ===")
    print("Registered model version:", MODEL_VERSION)
    print("Records checked:", len(sample))
    print("Input features:", sample.shape[1])
    print("Held-out test set: Not used")
    print("Predictions:", registered_predictions.tolist())

    print("\nAll registered model checks PASSED!")


if __name__ == "__main__":
    main()
