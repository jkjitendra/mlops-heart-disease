"""FastAPI service for the heart disease classifier."""

import logging
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from src.experiment_tracking import TRACKING_URI


MODEL_NAME = "heart-disease-classifier"
MODEL_ALIAS = "champion"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the approved MLflow model once during startup."""

    app.state.model = None
    app.state.model_info = None

    try:
        mlflow.set_tracking_uri(TRACKING_URI)

        client = MlflowClient(tracking_uri=TRACKING_URI)

        # Find the version currently assigned to champion.
        champion = client.get_model_version_by_alias(
            MODEL_NAME,
            MODEL_ALIAS,
        )

        if champion.status != "READY":
            raise RuntimeError(
                "Champion model version is not READY"
            )

        # Pin the exact version being loaded.
        version_uri = (
            f"models:/{MODEL_NAME}/{champion.version}"
        )

        model = mlflow.sklearn.load_model(version_uri)

        # Store metadata associated with the loaded model.
        app.state.model_info = {
            "model_name": MODEL_NAME,
            "alias": MODEL_ALIAS,
            "version": str(champion.version),
            "model_uri": MODEL_URI,
            "algorithm": type(
                model.named_steps["classifier"]
            ).__name__,
            "model_loaded": True,
            "source_run_id": champion.run_id,
        }

        app.state.model = model

        logger.info(
            "MLflow champion model loaded: %s (version %s)",
            MODEL_URI,
            champion.version,
        )

    except Exception:
        app.state.model = None
        app.state.model_info = None

        logger.exception(
            "Failed to load MLflow champion model"
        )

    yield

    # Release references during shutdown.
    app.state.model = None
    app.state.model_info = None


app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "Academic MLOps demonstration using "
        "a Random Forest model registered in MLflow."
    ),
    version="0.6.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request):
    """Report API and model availability."""

    model_loaded = request.app.state.model is not None

    return {
        "status": "ok" if model_loaded else "degraded",
        "service": "heart-disease-api",
        "model_loaded": model_loaded,
    }


@app.get("/ready")
def readiness(request: Request):
    """Report whether the API can serve predictions."""

    if request.app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail="model unavailable",
        )

    return {
        "status": "ready",
        "model_loaded": True,
        "model_version": (
            request.app.state.model_info["version"]
        ),
    }


@app.get("/model-info")
def model_info(request: Request):
    """Return metadata for the model loaded by this API."""

    if (
        request.app.state.model is None
        or request.app.state.model_info is None
    ):
        raise HTTPException(
            status_code=503,
            detail="model unavailable",
        )

    return request.app.state.model_info


class HeartDiseaseInput(BaseModel):
    """Validated Cleveland heart-disease input features."""

    age: float = Field(gt=0, le=120)
    sex: Literal[0, 1]
    cp: Literal[1, 2, 3, 4]
    trestbps: float = Field(gt=0, le=300)
    chol: float = Field(gt=0, le=1000)
    fbs: Literal[0, 1]
    restecg: Literal[0, 1, 2]
    thalach: float = Field(gt=0, le=300)
    exang: Literal[0, 1]
    oldpeak: float = Field(ge=0, le=10)
    slope: Literal[1, 2, 3]
    ca: Literal[0, 1, 2, 3] | None
    thal: Literal[3, 6, 7] | None

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        json_schema_extra={
            "example": {
                "age": 54,
                "sex": 1,
                "cp": 2,
                "trestbps": 130,
                "chol": 230,
                "fbs": 0,
                "restecg": 0,
                "thalach": 160,
                "exang": 0,
                "oldpeak": 1.0,
                "slope": 2,
                "ca": 0,
                "thal": 3,
            }
        },
    )


FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


@app.post("/predict")
def predict(patient: HeartDiseaseInput, request: Request):
    """Predict the binary heart-disease class."""

    model = request.app.state.model

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="model unavailable",
        )

    # Convert validated JSON fields into a model input.
    values = patient.model_dump()

    # Cleveland contains missing ca/thal values.
    # The saved pipeline already includes imputation.
    for feature in ("ca", "thal"):
        if values[feature] is None:
            values[feature] = float("nan")

    features = pd.DataFrame(
        [values],
        columns=FEATURE_COLUMNS,
    )

    try:
        predicted_class = int(
            model.predict(features)[0]
        )

        # Always select the probability for positive class 1.
        positive_index = list(model.classes_).index(1)

        probability = float(
            model.predict_proba(features)[0][positive_index]
        )

    except Exception as exc:
        logger.exception("Model prediction failed")

        raise HTTPException(
            status_code=500,
            detail="prediction failed",
        ) from exc

    return {
        "prediction": predicted_class,
        "probability": probability,
        "model_version": (
            request.app.state.model_info["version"]
        ),
        "model_alias": MODEL_ALIAS,
    }
