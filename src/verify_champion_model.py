"""Verify that the champion alias loads the expected model."""

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np

from mlflow.tracking import MlflowClient

from src.data import load_data, split_data
from src.experiment_tracking import TRACKING_URI


MODEL_NAME = "heart-disease-classifier"
VERSION = "1"
ALIAS = "champion"

VERSION_URI = f"models:/{MODEL_NAME}/{VERSION}"
CHAMPION_URI = f"models:/{MODEL_NAME}@{ALIAS}"

EXPECTED_RUN_ID = "7fdb1783f308430da79ab15b5d939078"

EXPECTED_SOURCE = (
    "models:/m-b853cd78dd7b4fbf8acfd01e414a817a"
)


def main():
    mlflow.set_tracking_uri(TRACKING_URI)

    client = MlflowClient(tracking_uri=TRACKING_URI)

    print("=== CHAMPION MODEL VERIFICATION ===")
    print("Version URI:", VERSION_URI)
    print("Champion URI:", CHAMPION_URI)

    # Verify that the alias points to the approved version.
    champion_version = client.get_model_version_by_alias(
        MODEL_NAME,
        ALIAS,
    )

    assert str(champion_version.version) == VERSION
    assert champion_version.status == "READY"
    assert champion_version.run_id == EXPECTED_RUN_ID
    assert champion_version.source == EXPECTED_SOURCE

    print("Champion alias metadata: VERIFIED")

    # Load the same model through two different references.
    version_model = mlflow.sklearn.load_model(
        VERSION_URI
    )

    champion_model = mlflow.sklearn.load_model(
        CHAMPION_URI
    )

    print("Champion model loading: PASS")

    # Use training data only.
    X, y = load_data()
    X_train, _, _, _ = split_data(X, y)

    sample = X_train.head(10).copy()

    # Verify class predictions.
    version_predictions = version_model.predict(sample)

    champion_predictions = champion_model.predict(
        sample
    )

    np.testing.assert_array_equal(
        version_predictions,
        champion_predictions,
    )

    print("Class predictions: IDENTICAL")

    # Verify predicted probabilities.
    version_probabilities = version_model.predict_proba(
        sample
    )

    champion_probabilities = (
        champion_model.predict_proba(sample)
    )

    np.testing.assert_allclose(
        version_probabilities,
        champion_probabilities,
        rtol=1e-10,
        atol=1e-12,
    )

    print("Prediction probabilities: IDENTICAL")

    # Verify the interface intended for API integration.
    pyfunc_model = mlflow.pyfunc.load_model(
        CHAMPION_URI
    )

    pyfunc_predictions = pyfunc_model.predict(sample)

    np.testing.assert_array_equal(
        champion_predictions,
        np.asarray(pyfunc_predictions),
    )

    print("Champion PyFunc predictions: IDENTICAL")

    print("\n=== VERIFICATION SUMMARY ===")
    print("Registered model:", MODEL_NAME)
    print("Champion version:", champion_version.version)
    print("Champion status:", champion_version.status)
    print("Records checked:", len(sample))
    print("Input features:", sample.shape[1])
    print("Held-out test set: Not used")
    print("Predictions:", champion_predictions.tolist())

    print("\nAll champion model checks PASSED!")


if __name__ == "__main__":
    main()
