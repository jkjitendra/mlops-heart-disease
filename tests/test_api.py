"""Automated tests for the heart-disease FastAPI service."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

from src.api import app


@pytest.fixture
def fake_model():
    """Build a predictable fake model for API tests."""

    model = MagicMock()

    model.named_steps = {
        "classifier": RandomForestClassifier()
    }

    model.classes_ = np.array([0, 1])

    model.predict.return_value = np.array([0])

    model.predict_proba.return_value = np.array([
        [0.8, 0.2]
    ])

    return model


@pytest.fixture
def client(fake_model):
    """Start FastAPI with mocked MLflow dependencies."""

    champion = SimpleNamespace(
        version="1",
        status="READY",
        run_id="test-run-id",
    )

    with (
        patch("src.api.MlflowClient") as client_class,
        patch(
            "src.api.mlflow.sklearn.load_model",
            return_value=fake_model,
        ),
    ):
        client_class.return_value.get_model_version_by_alias.return_value = (
            champion
        )

        # Using the context manager executes FastAPI lifespan.
        with TestClient(app) as test_client:
            yield test_client


def test_health_when_model_loaded(client):
    """A healthy API reports a loaded model."""

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "ok",
        "service": "heart-disease-api",
        "model_loaded": True,
    }


def test_model_info(client):
    """Model information reflects the loaded champion."""

    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == "heart-disease-classifier"
    assert data["alias"] == "champion"
    assert data["version"] == "1"
    assert data["algorithm"] == "RandomForestClassifier"
    assert data["model_loaded"] is True
    assert data["source_run_id"] == "test-run-id"


def test_openapi_documentation(client):
    """All three API endpoints appear in the schema."""

    response = client.get("/openapi.json")

    assert response.status_code == 200

    paths = response.json()["paths"]

    assert "/health" in paths
    assert "/model-info" in paths
    assert "/predict" in paths
    assert "/ready" in paths


def test_api_when_model_unavailable():
    """API remains responsive when the registry fails."""

    with patch(
        "src.api.MlflowClient",
        side_effect=RuntimeError(
            "Simulated MLflow registry outage"
        ),
    ):
        with TestClient(app) as test_client:

            health = test_client.get("/health")

            assert health.status_code == 200
            assert health.json()["status"] == "degraded"
            assert health.json()["model_loaded"] is False

            readiness = test_client.get("/ready")

            assert readiness.status_code == 503
            assert readiness.json()["detail"] == (
                "model unavailable"
            )

            info = test_client.get("/model-info")

            assert info.status_code == 503
            assert info.json()["detail"] == "model unavailable"

            valid_request = {
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

            prediction = test_client.post(
                "/predict",
                json=valid_request,
            )

            assert prediction.status_code == 503
            assert prediction.json()["detail"] == (
                "model unavailable"
            )


# Reusable valid request for prediction tests.
VALID_PATIENT = {
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


def test_predict_valid_patient(client, fake_model):
    """A valid request returns a prediction and probability."""

    response = client.post(
        "/predict",
        json=VALID_PATIENT,
    )

    assert response.status_code == 200

    data = response.json()

    assert data == {
        "prediction": 0,
        "probability": pytest.approx(0.2),
        "model_version": "1",
        "model_alias": "champion",
    }

    fake_model.predict.assert_called_once()
    fake_model.predict_proba.assert_called_once()

    # Ensure all 13 features reach the model.
    model_input = fake_model.predict.call_args.args[0]

    assert model_input.shape == (1, 13)


def test_predict_with_missing_optional_values(
    client,
    fake_model,
):
    """Null ca and thal values reach preprocessing as NaN."""

    patient = {
        **VALID_PATIENT,
        "ca": None,
        "thal": None,
    }

    response = client.post(
        "/predict",
        json=patient,
    )

    assert response.status_code == 200

    model_input = fake_model.predict.call_args.args[0]

    assert np.isnan(model_input.loc[0, "ca"])
    assert np.isnan(model_input.loc[0, "thal"])


@pytest.mark.parametrize(
    "feature,invalid_value",
    [
        ("age", -5),
        ("sex", 3),
        ("cp", 9),
        ("chol", -100),
        ("oldpeak", 11),
        ("thal", 5),
        ("age", "invalid"),
    ],
)
def test_predict_rejects_invalid_values(
    client,
    fake_model,
    feature,
    invalid_value,
):
    """Invalid inputs must be rejected before inference."""

    patient = {
        **VALID_PATIENT,
        feature: invalid_value,
    }

    response = client.post(
        "/predict",
        json=patient,
    )

    assert response.status_code == 422

    errors = response.json()["detail"]

    assert any(
        error["loc"][-1] == feature
        for error in errors
    )

    fake_model.predict.assert_not_called()


def test_predict_rejects_missing_field(client, fake_model):
    """Missing mandatory features return HTTP 422."""

    patient = VALID_PATIENT.copy()
    patient.pop("age")

    response = client.post(
        "/predict",
        json=patient,
    )

    assert response.status_code == 422

    fake_model.predict.assert_not_called()


def test_predict_rejects_extra_field(client, fake_model):
    """Unexpected request fields are rejected."""

    patient = {
        **VALID_PATIENT,
        "unknown_field": 123,
    }

    response = client.post(
        "/predict",
        json=patient,
    )

    assert response.status_code == 422

    fake_model.predict.assert_not_called()


def test_predict_failure_returns_500(client, fake_model):
    """An inference error returns a controlled API error."""

    fake_model.predict.side_effect = RuntimeError(
        "Simulated internal model failure"
    )

    response = client.post(
        "/predict",
        json=VALID_PATIENT,
    )

    assert response.status_code == 500

    assert response.json() == {
        "detail": "prediction failed"
    }


def test_positive_probability_class_mapping(
    client,
    fake_model,
):
    """Probability must correspond to class 1, not position."""

    # Deliberately reverse class order.
    fake_model.classes_ = np.array([1, 0])

    fake_model.predict.return_value = np.array([1])

    fake_model.predict_proba.return_value = np.array([
        [0.3, 0.7]
    ])

    response = client.post(
        "/predict",
        json=VALID_PATIENT,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == 1
    assert data["probability"] == pytest.approx(0.3)



def test_ready_when_model_loaded(client):
    """The API is ready when its model has loaded."""

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model_loaded": True,
        "model_version": "1",
    }


def test_mlflow_tracking_uri_environment_override(
    monkeypatch,
    fake_model,
):
    """An environment variable overrides the local MLflow URI."""

    test_uri = "http://mlflow:5050"

    monkeypatch.setenv("MLFLOW_TRACKING_URI", test_uri)

    champion = SimpleNamespace(
        version="1",
        status="READY",
        run_id="test-run-id",
    )

    with (
        patch("src.api.MlflowClient") as client_class,
        patch("src.api.mlflow.set_tracking_uri") as set_uri,
        patch(
            "src.api.mlflow.sklearn.load_model",
            return_value=fake_model,
        ),
    ):
        client_class.return_value.get_model_version_by_alias.return_value = (
            champion
        )

        with TestClient(app) as test_client:
            response = test_client.get("/ready")

            assert response.status_code == 200

        set_uri.assert_called_once_with(test_uri)

        client_class.assert_called_once_with(
            tracking_uri=test_uri
        )
