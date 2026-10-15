"""Register the verified Random Forest model in MLflow."""

import mlflow
from mlflow.tracking import MlflowClient

from src.experiment_tracking import TRACKING_URI


MODEL_ID = "m-b853cd78dd7b4fbf8acfd01e414a817a"

MODEL_URI = f"models:/{MODEL_ID}"

SOURCE_RUN_ID = "7fdb1783f308430da79ab15b5d939078"

REGISTERED_MODEL_NAME = "heart-disease-classifier"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)

    client = MlflowClient(tracking_uri=TRACKING_URI)

    print("=== MLFLOW MODEL REGISTRATION ===")
    print("Source Model URI:", MODEL_URI)
    print("Registered Model Name:", REGISTERED_MODEL_NAME)

    # Check that the original logged model exists.
    logged_model = client.get_logged_model(MODEL_ID)

    if logged_model.source_run_id != SOURCE_RUN_ID:
        raise RuntimeError(
            "The logged model does not match the expected run."
        )

    # Avoid creating another version when rerunning this script.
    existing_versions = client.search_model_versions(
        filter_string=f"name = '{REGISTERED_MODEL_NAME}'"
    )

    for existing in existing_versions:
        if (
            existing.source == MODEL_URI
            and existing.run_id == SOURCE_RUN_ID
        ):
            if existing.status != "READY":
                raise RuntimeError(
                    f"Existing version is not ready: {existing.status}"
                )
            print("\nModel is already registered.")
            print("Version:", existing.version)
            print(
                "Registered Model URI:",
                f"models:/{REGISTERED_MODEL_NAME}/{existing.version}",
            )
            print("Status:", existing.status)
            return

    # Register the existing model without retraining.
    model_version = mlflow.register_model(
        model_uri=MODEL_URI,
        name=REGISTERED_MODEL_NAME,
        tags={
            "algorithm": "RandomForestClassifier",
            "dataset": "UCI Cleveland Heart Disease",
            "validation": "packaged_model_verified",
            "intended_use": "academic_demonstration",
        },
    )

    if model_version.status != "READY":
        raise RuntimeError(
            f"Unexpected model status: {model_version.status}"
        )

    if (
        model_version.source != MODEL_URI
        or model_version.run_id != SOURCE_RUN_ID
    ):
        raise RuntimeError(
            "Registered model source or run ID does not match."
        )

    print("\n=== REGISTRATION SUCCESSFUL ===")
    print("Registered Name:", model_version.name)
    print("Version:", model_version.version)
    print("Status:", model_version.status)
    print("Source Run ID:", model_version.run_id)
    print(
        "Registered Model URI:",
        f"models:/{model_version.name}/{model_version.version}",
    )

    print("\nModel registration completed successfully!")


if __name__ == "__main__":
    main()
