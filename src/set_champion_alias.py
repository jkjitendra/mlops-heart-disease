"""Assign and verify the MLflow champion model alias."""

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

from src.experiment_tracking import TRACKING_URI


MODEL_NAME = "heart-disease-classifier"
MODEL_VERSION = "1"
ALIAS = "champion"

EXPECTED_SOURCE = (
    "models:/m-b853cd78dd7b4fbf8acfd01e414a817a"
)

EXPECTED_RUN_ID = "7fdb1783f308430da79ab15b5d939078"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)

    client = MlflowClient(tracking_uri=TRACKING_URI)

    print("=== MLFLOW CHAMPION ALIAS ===")

    # Verify the model version before assigning an alias.
    version = client.get_model_version(
        MODEL_NAME,
        MODEL_VERSION,
    )

    if version.status != "READY":
        raise RuntimeError("Model version is not READY")

    if (
        version.source != EXPECTED_SOURCE
        or version.run_id != EXPECTED_RUN_ID
    ):
        raise RuntimeError(
            "Model source verification failed"
        )

    print("Registered Version:", version.version)
    print("Source verification: PASS")

    # Check whether the alias already exists.
    try:
        current = client.get_model_version_by_alias(
            MODEL_NAME,
            ALIAS,
        )
    except MlflowException as exc:
        missing_alias = (
            exc.error_code in {
                "RESOURCE_DOES_NOT_EXIST",
                "INVALID_PARAMETER_VALUE",
            }
            and (
                f"registered model alias {ALIAS} not found"
                in str(exc).lower()
            )
        )

        if not missing_alias:
            raise

        current = None

    if current is not None:
        if str(current.version) != MODEL_VERSION:
            raise RuntimeError(
                f"Alias already points to Version "
                f"{current.version}; refusing to overwrite"
            )

        print("Champion alias already configured.")
    else:
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias=ALIAS,
            version=MODEL_VERSION,
        )

        print("Champion alias assigned successfully.")

    # Independently verify the alias destination.
    champion = client.get_model_version_by_alias(
        MODEL_NAME,
        ALIAS,
    )

    assert str(champion.version) == MODEL_VERSION
    assert champion.status == "READY"
    assert champion.source == EXPECTED_SOURCE
    assert champion.run_id == EXPECTED_RUN_ID

    print("\n=== ALIAS VERIFICATION ===")
    print("Registered Model:", MODEL_NAME)
    print("Alias:", ALIAS)
    print("Version:", champion.version)
    print("Status:", champion.status)
    print(
        "Champion Model URI:",
        f"models:/{MODEL_NAME}@{ALIAS}",
    )

    print("\nChampion alias verification PASSED!")


if __name__ == "__main__":
    main()
