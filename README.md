# Heart Disease Prediction — End-to-End MLOps

## 1. Project Overview

This project was developed for the BITS Pilani MLOps
Experimental Learning Assignment 1.

It demonstrates an end-to-end machine learning workflow:
data preparation, exploratory analysis, feature engineering,
model training and tuning, MLflow experiment tracking,
model registration, automated CI, Docker containerization,
local Kubernetes deployment, and API monitoring.

The application predicts the binary presence or absence of
heart disease using the UCI Cleveland dataset.

**Disclaimer:** This is an academic demonstration.
It is not intended for clinical diagnosis or medical decisions.

## 2. Dataset

Source: UCI Machine Learning Repository

Dataset: Heart Disease — Processed Cleveland

https://archive.ics.uci.edu/dataset/45/heart+disease

- Total records: 303
- Input features: 13
- Target: `heart_disease`
- Class 0: No heart disease
- Class 1: Heart disease present
- Training records: 242
- Held-out testing records: 61
- Random state: 42

Original target values 1–4 are mapped to binary class 1.
Original target value 0 remains class 0.

Missing values are retained during data preparation and
handled using train-fitted preprocessing inside the
scikit-learn Pipelines to avoid data leakage.

## 3. Technology Stack

| Area | Tools |
|---|---|
| Data processing | Python, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine learning | Scikit-learn |
| Experiment tracking | MLflow |
| Model packaging | MLflow and skops |
| API | FastAPI, Uvicorn |
| Automated tests | Pytest |
| Linting | Ruff |
| CI | GitHub Actions |
| Containerization | Docker |
| Orchestration | Colima, K3s Kubernetes |
| Monitoring | API logs and in-memory metrics dashboard |
| Version control | Git and GitHub |

## 4. Model Development

Four classifier families were evaluated:

1. Logistic Regression
2. Random Forest
3. Support Vector Machine (RBF)
4. Gradient Boosting

Eight baseline and tuned configurations were tracked
and compared using MLflow.

Model comparison used five-fold stratified cross-validation
on training data only.

The primary selection metric was ROC-AUC.

### Selected Model

**Tuned Random Forest**

- Number of trees: 200
- Maximum depth: 5
- Minimum samples per leaf: 3
- Class weight: balanced
- Random state: 42
- Mean cross-validation ROC-AUC: 0.9044

The advantage over tuned Logistic Regression was small.
The results do not establish statistically significant
superiority over the alternative models.

### Held-out Test Results

| Metric | Value |
|---|---:|
| Accuracy | 0.8852 |
| Precision | 0.8387 |
| Recall | 0.9286 |
| F1-score | 0.8814 |
| ROC-AUC | 0.9556 |

These results were obtained on 61 held-out test records.

Detailed results are available in:

- `reports/model_selection.md`
- `reports/mlflow_experiment_summary.md`
- `reports/test_metrics.json`
- `reports/figures/`

## 5. System Architecture

See [Architecture Diagram](docs/architecture.md).

The architecture separates:

- Dataset preparation, training, evaluation and MLflow
- GitHub Actions continuous integration
- Docker and local Kubernetes serving
- API logging and monitoring

CI validates the application but does not automatically
deploy a new model or container to Kubernetes.

## 6. Project Setup

### Requirements

- Python 3.12
- Git
- Docker
- Kubernetes cluster for deployment
- MLflow tracking server for model registration and serving

Clone the repository:

```bash
git clone https://github.com/jkjitendra/mlops-heart-disease.git
cd mlops-heart-disease
```

The repository is private and requires authorized access.

Create a Python environment on macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-serving.txt
python -m pip install ruff==0.13.3
```

The original development environment uses an APFS workspace
on an external SSD. That setup is machine-specific and is
not required for general Python development.

On the original development machine, the workspace is
opened with:

```bash
source ~/open-mlops.sh
```

## 7. Data Preparation and Model Training

The original raw dataset and prepared dataset are
included in the repository.

Regenerate the prepared dataset:

```bash
python -m scripts.prepare_data
```

Run baseline model training and cross-validation:

```bash
python -m src.train
```

Run automated tests and linting:

```bash
python -m pytest -q
python -m ruff check src tests scripts
```

The existing test suite contains 22 tests.

The baseline training script evaluates two models:
Logistic Regression and Random Forest.

The expanded four-model comparison is documented in
the MLflow experiment report.

## 8. MLflow Tracking and Model Registry

Start MLflow from the repository root:

```bash
mlflow server \
  --backend-store-uri "sqlite:///$PWD/mlflow.db" \
  --artifacts-destination "$PWD/mlartifacts" \
  --host 127.0.0.1 \
  --port 5050 \
  --workers 1 \
  --allowed-hosts "localhost:*,127.0.0.1:*,host.lima.internal:5050"
```

MLflow UI:

http://127.0.0.1:5050

The current deployment uses:

- Registered model: `heart-disease-classifier`
- Registered version: `1`
- Model alias: `champion`
- Model URI: `models:/heart-disease-classifier@champion`

The full fitted preprocessing and Random Forest pipeline
is packaged using MLflow and skops.

Model packaging script:

```bash
python -m src.package_model
```

**Reproducibility note:** The existing registration and
champion-alias scripts reference specific MLflow model IDs
and run IDs generated during the original experiments.

A fresh checkout can prepare data, train models and run
tests, but it does not automatically recreate the original
MLflow registry or champion alias.

For a new MLflow registry, update the source identifiers
in the registration and verification scripts to match
newly created MLflow artifacts before executing them.

The existing serving API requires a reachable MLflow
server with the registered champion model available.

Local MLflow databases and artifacts are not committed
to Git.

## 9. Docker Containerization

The serving image uses Python 3.12 and installs the pinned
dependencies from `requirements-serving.txt`.

It includes:

- `src/api.py`
- `src/monitoring.py`
- Model loading from MLflow
- FastAPI inference and monitoring endpoints

Build the monitoring-enabled Docker image:

```bash
docker build -t heart-disease-api:0.2 .
```

The application uses port 8000 inside the container.

For containerized execution, the environment variable
`MLFLOW_TRACKING_URI` must point to a reachable MLflow
tracking server containing the champion model.

## 10. Kubernetes Deployment

The application runs in a local Kubernetes cluster using
Colima and K3s.

The original development machine uses an existing
Colima profile named `mlops`.

Configure the external Colima workspace:

```bash
export COLIMA_HOME="/Volumes/MLOpsWorkspace/.colima"
export COLIMA_CACHE_HOME="/Volumes/MLOpsWorkspace/.colima-cache"
```

Start the existing cluster:

```bash
colima start mlops --activate=false
```

Build the image in that Docker environment:

```bash
docker --context colima-mlops build \
  -t heart-disease-api:0.2 .
```

Apply the deployment configuration:

```bash
kubectl --context colima-mlops apply \
  -f k8s/heart-disease-api.yaml
```

Check the deployment:

```bash
kubectl --context colima-mlops rollout status \
  deployment/heart-disease-api \
  -n heart-disease \
  --timeout=180s
```

The supplied manifest expects the image to be available
locally and MLflow to be reachable from the container at
`http://host.lima.internal:5050`.

This procedure assumes the Colima/K3s profile has already
been provisioned and configured.

The deployed Kubernetes LoadBalancer is accessible from
the original Mac host at:

http://127.0.0.1:8088

This is a local endpoint, not a public cloud URL.

## 11. API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | API and model health |
| `/ready` | GET | Model readiness |
| `/model-info` | GET | Registered model metadata |
| `/predict` | POST | Heart disease prediction |
| `/metrics` | GET | Request counts and latency |
| `/monitor` | GET | Browser monitoring dashboard |
| `/docs` | GET | FastAPI Swagger documentation |

Sample prediction request:

```bash
curl --noproxy '*' \
  -X POST http://127.0.0.1:8088/predict \
  -H 'Content-Type: application/json' \
  -d '{"age":54,"sex":1,"cp":2,"trestbps":130,"chol":230,"fbs":0,"restecg":0,"thalach":160,"exang":0,"oldpeak":1.0,"slope":2,"ca":0,"thal":3}'
```

The response includes the predicted class, positive-class
probability, model version and model alias.

## 12. Continuous Integration

GitHub Actions runs automatically on pushes and pull
requests targeting `main`.

The CI workflow contains three jobs:

1. Python Tests: Ruff linting and Pytest
2. Model Training: Baseline five-fold cross-validation
3. Docker Build: Container build validation

The Model Training job uploads a downloadable artifact
named `model-training-results`.

The CI workflow does not promote the registered model,
publish a production image or deploy to Kubernetes.

Workflow definition: `.github/workflows/ci.yml`

## 13. Monitoring and Reliability

The API provides an in-memory monitoring dashboard:

http://127.0.0.1:8088/monitor

The dashboard displays:

- Prediction request count
- Successful request count
- HTTP 4xx and 5xx errors
- Average request latency
- p95 request latency

Uvicorn logs HTTP requests and response statuses.
Kubernetes logs can be inspected using:

```bash
kubectl --context colima-mlops logs \
  deployment/heart-disease-api \
  -n heart-disease
```

### Observed Latency Baseline

200 sequential prediction requests, following five
warm-up requests:

| Metric | Result |
|---|---:|
| Successful requests | 200/200 |
| Average latency | 27.93 ms |
| p50 | 22.75 ms |
| p95 | 51.32 ms |
| p99 | 56.14 ms |

A separate reliability test observed 100 successful valid
requests and 10 correctly rejected invalid requests.

During a controlled MLflow outage, the existing Pod
continued inference using its already-loaded model.

An isolated new Pod did not become ready during a
42-second observation while MLflow was unavailable.

Monitoring counters reset when the application process
restarts. They are not stored in an external database.

Details: `docs/monitoring_baseline.md`

## 14. Reports and Evidence

- `reports/figures/` — EDA and model evaluation figures
- `reports/model_selection.md` — Model-selection rationale
- `reports/mlflow_experiment_summary.md` — MLflow comparison
- `reports/test_metrics.json` — Held-out test metrics
- `docs/architecture.md` — System architecture
- `docs/monitoring_baseline.md` — Reliability experiments
- `docs/screenshots/` — Kubernetes, API, monitoring and CI evidence

## 15. Limitations

- The Cleveland dataset contains only 303 records.
- Reported metrics do not establish clinical validity.
- The Kubernetes deployment is local, not publicly hosted.
- CI training is separate from production model promotion.
- The registry depends on local MLflow state and artifacts.
- Some registration scripts use experiment-specific IDs.
- API monitoring metrics are in memory and are not persistent.
- Latency tests used sequential rather than concurrent traffic.

## Repository

https://github.com/jkjitendra/mlops-heart-disease

This repository is intended for academic MLOps learning.
