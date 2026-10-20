# Heart Disease Prediction — End-to-End MLOps

## 1. Project Overview

This project was developed for the **BITS Pilani MLOps Experimental Learning Assignment 1**.

It demonstrates an end-to-end machine learning operations workflow: data acquisition, exploratory data analysis (EDA), preprocessing, model training and evaluation, MLflow experiment tracking and model registration, GitHub Actions continuous integration (CI), Docker containerization, Kubernetes deployment, and API monitoring.

The application predicts the binary presence or absence of heart disease using the **UCI Processed Cleveland** dataset.

**Disclaimer:** This is an academic demonstration, not a medical diagnostic system. Its outputs must not be used for clinical decisions.

## 2. Dataset

**Source:** [UCI Machine Learning Repository — Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)

| Property          | Value                                |
| ----------------- | ------------------------------------ |
| Records           | 303                                  |
| Input features    | 13                                   |
| Binary target     | `heart_disease`                      |
| Class 0           | No heart disease recorded            |
| Class 1           | Heart disease present in the dataset |
| Training set      | 242 records                          |
| Held-out test set | 61 records                           |
| Random state      | 42                                   |

The original target value `0` maps to Class 0; values `1`–`4` map to Class 1. Missing values remain in the prepared data and are handled within scikit-learn preprocessing pipelines fitted on training data, avoiding leakage from the held-out test set.

## 3. Technology Stack

| Area                                   | Technology                                    |
| -------------------------------------- | --------------------------------------------- |
| Data preparation and analysis          | Python, Pandas, NumPy, Matplotlib, Seaborn    |
| Model development                      | scikit-learn                                  |
| Experiment tracking and model registry | MLflow                                        |
| Model serialization                    | MLflow, skops                                 |
| Prediction API                         | FastAPI, Uvicorn                              |
| Testing and linting                    | Pytest, Ruff                                  |
| Continuous integration                 | GitHub Actions                                |
| Containerization                       | Docker                                        |
| Kubernetes on **Mac**                  | Colima with K3s                               |
| Kubernetes on **Linux (VM)**           | Minikube with Docker driver                   |
| Monitoring                             | FastAPI metrics dashboard and Kubernetes logs |
| Version control                        | Git, GitHub                                   |

## 4. Model Development and Results

Four classifier families were evaluated: **Logistic Regression, Random Forest, Support Vector Machine (RBF), and Gradient Boosting**. Eight baseline/tuned configurations were compared using five-fold stratified cross-validation on the training set; the primary selection metric was ROC-AUC (area under the receiver operating characteristic curve).

### Selected model — Tuned Random Forest

| Setting                       | Value      |
| ----------------------------- | ---------- |
| Number of trees               | 200        |
| Maximum depth                 | 5          |
| Minimum samples per leaf      | 3          |
| Class weight                  | `balanced` |
| Random state                  | 42         |
| Mean cross-validation ROC-AUC | **0.9044** |

The improvement over tuned Logistic Regression was small and does not establish statistically significant superiority.

### Held-out test results

| Metric    |  Value |
| --------- | -----: |
| Accuracy  | 0.8852 |
| Precision | 0.8387 |
| Recall    | 0.9286 |
| F1-score  | 0.8814 |
| ROC-AUC   | 0.9556 |

These metrics were measured on **61 held-out records**. See `reports/model_selection.md`, `reports/mlflow_experiment_summary.md`, `reports/test_metrics.json`, and `reports/figures/`.

## 5. System Architecture

See [System Architecture](docs/architecture.md).

The project separates model development and MLflow tracking, automated CI checks, and Docker/Kubernetes model serving. **GitHub Actions validates the code and build but does not automatically register models or deploy Kubernetes resources.**

## 6. Prerequisites — Mac and Linux (VM)

Both environments require:

- **Python 3.12**, pip, and Git
- An authorized GitHub account if the repository is private
- A terminal with access to the repository
- Docker Engine/CLI for building the serving image
- An MLflow server and a registered model aliased as `champion` for inference
- A working Kubernetes environment for deploying the API
- Sufficient disk, CPU and RAM for model training, MLflow, the image build and Kubernetes

The Python dependencies are listed in `requirements.txt` and `requirements-serving.txt`. Ruff is used for linting.

### Mac prerequisites

Install the required command-line tools, using Homebrew if desired:

```bash
brew install git python@3.12 colima docker kubectl
```

Colima supplies the Docker runtime and K3s Kubernetes environment. Check existing installations before installing or replacing anything.

### Linux (VM) prerequisites

Install **Python 3.12, Git, Docker Engine, and Minikube** using packages appropriate for the VM distribution and CPU architecture. Verify that Docker works for your user and that the VM can run the Minikube Docker driver. For installation guidance, see:

- [Docker Engine installation](https://docs.docker.com/engine/install/)
- [Minikube installation](https://minikube.sigs.k8s.io/docs/start/)

Confirm the tools before proceeding:

```bash
python3.12 --version
git --version
docker version
docker info
minikube version
```

On hosted VMs, use a **persistent workspace** for the repository, `mlflow.db`, and `mlartifacts/` if the provider's system disk can be replaced. `~/workspace` is used below as an example; persistence depends on the VM provider's storage configuration.

## 7. Clone, Install, Prepare Data, and Test — Both Environments

Run these commands on **Mac or Linux (VM)** from the directory where you want the repository stored:

```bash
git clone https://github.com/jkjitendra/mlops-heart-disease.git
cd mlops-heart-disease

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-serving.txt
python -m pip install ruff==0.13.3
```

If using an **existing Conda environment** instead of a new Python `venv`, activate it using `conda activate <environment-name-or-path>`. A Conda environment does not necessarily contain `.venv/bin/activate`.

The repository includes the raw and prepared datasets. To regenerate the prepared data, run baseline training, and check the code:

```bash
python -m scripts.prepare_data
python -m src.train
python -m pytest -q
python -m ruff check src tests scripts
```

The test suite contains **22 tests**. `src.train` runs the baseline comparison of Logistic Regression and Random Forest; the four-family/eight-configuration comparison is documented in the MLflow experiment report and captured evidence. **Running the baseline training command alone does not recreate the full eight-run comparison.**

## 8. MLflow Tracking, Packaging, and Model Registry

The serving API loads the registered model using:

```text
models:/heart-disease-classifier@champion
```

**Important for fresh clones:** `mlflow.db` and `mlartifacts/` are intentionally excluded from Git. A new clone does **not** contain the original experiment runs, registered model versions, or the champion alias. Start the MLflow server, package a model, and register an available model as `champion` before deploying the API. Existing registry state can instead be restored from a compatible backup of both the backend database and its artifacts.

### 8.1 Start MLflow on Mac

Open a separate terminal, enter the repository directory, and activate its Python environment. Then run:

```bash
mlflow server \
  --backend-store-uri "sqlite:///$PWD/mlflow.db" \
  --artifacts-destination "$PWD/mlartifacts" \
  --host 0.0.0.0 \
  --port 5050 \
  --workers 1 \
  --allowed-hosts "localhost:*,127.0.0.1:*,host.lima.internal:*"
```

This binding is required by the supplied **Mac** Kubernetes manifest so the pod can reach MLflow through `host.lima.internal`. Run it only on a **trusted, restricted network**: MLflow is not configured with authentication in this demonstration. Do not expose port 5050 to the public internet.

### 8.2 Start MLflow on Linux (VM)

Open a separate terminal, enter the repository directory, and activate its Python environment. Then run:

```bash
mlflow server \
  --backend-store-uri "sqlite:///$PWD/mlflow.db" \
  --artifacts-destination "$PWD/mlartifacts" \
  --host 127.0.0.1 \
  --port 5050 \
  --workers 1 \
  --allowed-hosts "localhost:*,127.0.0.1:*"
```

On **Linux (VM)**, the Kubernetes pod reaches MLflow through the repository's private Minikube bridge, configured in Section 10. This keeps the MLflow server bound to loopback rather than a public VM interface. The bridge is for a trusted academic environment, not a hardened production proxy.

### 8.3 Verify MLflow — Both environments

Leave the MLflow terminal running. In another terminal, activate the project Python environment and run:

```bash
curl -f http://127.0.0.1:5050/health
export MLFLOW_TRACKING_URI=http://127.0.0.1:5050
```

Open the MLflow UI at **http://127.0.0.1:5050** in a browser running on the same system as the MLflow server, or use an SSH tunnel if accessing a VM remotely.

### 8.4 Package and register the model — First-time setup

Once data preparation is complete and MLflow is running:

```bash
python -m src.package_model
```

In MLflow, open the `heart-disease-model-packaging` experiment and locate the newly logged model. Copy its **model URI**, of the form `models:/m-...`. To register that specific logged model, run the following in the terminal where `MLFLOW_TRACKING_URI` is set:

```bash
# Replace the placeholder with the model URI shown in MLflow.
export PACKAGED_MODEL_URI='models:/m-REPLACE_WITH_YOUR_MODEL_ID'

python - <<'PY'
import os
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
model_version = mlflow.register_model(
    model_uri=os.environ['PACKAGED_MODEL_URI'],
    name='heart-disease-classifier',
    await_registration_for=300,
)
client = MlflowClient()
client.set_registered_model_alias(
    'heart-disease-classifier', 'champion', model_version.version
)
selected = client.get_model_version_by_alias(
    'heart-disease-classifier', 'champion'
)
print(f'Champion: {selected.name}, version {selected.version}, status {selected.status}')
PY
```

The placeholder **must** be replaced with a real URI from the current MLflow instance. The registration creates the next available version; **version 1 applies only to a newly registered model**. Some additional registration/verification scripts in the repository use experiment-specific IDs and must be updated if run against a new MLflow database. This packaging workflow does not automatically reproduce the original eight experimental runs.

To verify an already registered champion:

```bash
python - <<'PY'
from mlflow.tracking import MlflowClient
client = MlflowClient(tracking_uri='http://127.0.0.1:5050')
model = client.get_model_version_by_alias('heart-disease-classifier', 'champion')
print('Model:', model.name, '| Version:', model.version, '| Status:', model.status)
PY
```

## 9. Docker Image — Both Environments

The Dockerfile uses Python 3.12 and `requirements-serving.txt`. It packages the FastAPI service and monitoring code; the container loads the champion model from MLflow at startup. The container listens on **port 8000**.

Run from the repository root in the Docker environment you intend to deploy to:

```bash
docker build -t heart-disease-api:0.2 .
docker image ls heart-disease-api:0.2
docker image inspect heart-disease-api:0.2 --format '{{.Architecture}} {{.Os}}'
```

A fresh Docker environment does not contain images built in a previous environment. Build or load the image again when switching Docker profiles or VM instances.

## 10. Kubernetes Deployment

Choose **one** of the following guides. The two environments use different Kubernetes manifests because MLflow networking is different.

### 10.A — Mac: Colima + K3s

**1. Start Kubernetes.** A new Colima profile can be provisioned as follows:

```bash
colima start --profile mlops --kubernetes --cpu 4 --memory 6 --disk 20
kubectl config use-context colima-mlops
docker context use colima-mlops
kubectl get nodes
```

For an existing `mlops` profile, `colima start --profile mlops` is sufficient; confirm Kubernetes is enabled using `colima status --profile mlops`. Avoid starting only the `default` profile while using the `colima-mlops` Kubernetes context.

**2. Verify Mac-to-MLflow connectivity and build the image in the correct Docker context.** Make sure the Mac MLflow server from Section 8.1 is running:

```bash
colima ssh --profile mlops -- curl -f http://host.lima.internal:5050/health
docker context show
docker build -t heart-disease-api:0.2 .
```

The verified Colima/K3s configuration uses Docker as its Kubernetes runtime, allowing K3s to use the image from that Docker context directly. Check with `kubectl get nodes -o wide` and `kubectl get node colima-mlops -o jsonpath='{.status.nodeInfo.containerRuntimeVersion}{"\n"}'` if unsure.

**3. Deploy and check readiness.** The Mac manifest uses `MLFLOW_TRACKING_URI=http://host.lima.internal:5050`:

```bash
kubectl apply -f k8s/heart-disease-api.yaml
kubectl rollout status deployment/heart-disease-api -n heart-disease --timeout=180s
kubectl get deployment,pods,services -n heart-disease
kubectl logs deployment/heart-disease-api -n heart-disease --tail=40
```

Expect one ready deployment and one running pod, with a log message showing the champion model was loaded. The manifest defines health probes, CPU/memory limits, a ClusterIP service and a LoadBalancer service on **8088**.

**4. Access the API from Mac.** In the verified Colima setup, the LoadBalancer is reachable at `http://127.0.0.1:8088`. Confirm it rather than assuming the address will be identical on every machine:

```bash
curl -f http://127.0.0.1:8088/health
curl -f http://127.0.0.1:8088/model-info
```

If port 8088 is not reachable, use a separate terminal and keep it running:

```bash
kubectl -n heart-disease port-forward svc/heart-disease-api 8088:8000 --address 127.0.0.1
```

The port-forward requires port 8088 to be available on the Mac.

### 10.B — Linux (VM): Minikube + Docker

**1. Start Minikube.** Confirm Docker is working, then initialize the Docker-driver cluster:

```bash
minikube start --driver=docker --cpus=2 --memory=4096
minikube status
minikube kubectl -- get nodes
```

**2. Build the serving image and load it into Minikube.** Run from the repository root:

```bash
docker build -t heart-disease-api:0.2 .
minikube image load heart-disease-api:0.2
```

**3. Start the private MLflow bridge.** Keep the **Linux (VM)** MLflow server from Section 8.2 running. The supplied `scripts/mlflow_vm_bridge.py` forwards requests from the Minikube bridge interface on port **5051** to MLflow at `127.0.0.1:5050`.

```bash
# IP of the Linux (VM) host as seen from Minikube
BRIDGE_IP="$(minikube ssh -- getent hosts host.minikube.internal | awk 'NR==1 {print $1}')"
echo "Minikube host bridge: $BRIDGE_IP"

# Start the bridge from the repository directory with the project Python environment
MLFLOW_BRIDGE_HOST="$BRIDGE_IP" nohup .venv/bin/python -u scripts/mlflow_vm_bridge.py \
  > /tmp/mlflow_vm_bridge.log 2>&1 < /dev/null &

curl -f "http://${BRIDGE_IP}:5051/health"
```

If the bridge IP is empty or the health check fails, stop and check Minikube host networking, the MLflow server and `/tmp/mlflow_vm_bridge.log`. The bridge intentionally listens on the Docker-facing interface; **do not expose port 5051 publicly**. If using Conda instead of `.venv/bin/python`, substitute the active environment's `python` executable.

**4. Create a runtime-specific manifest and deploy.** The tracked Linux manifest uses the example address `192.168.49.1`; substitute the discovered bridge address without changing the checked-in source file:

```bash
sed "s#http://192.168.49.1:5051#http://${BRIDGE_IP}:5051#g" \
  k8s/heart-disease-api-prayogshala.yaml > /tmp/heart-disease-api-vm.yaml

minikube kubectl -- apply -f /tmp/heart-disease-api-vm.yaml
minikube kubectl -- -n heart-disease rollout status deployment/heart-disease-api --timeout=180s
minikube kubectl -- -n heart-disease get deployments,pods,services
minikube kubectl -- -n heart-disease logs deployment/heart-disease-api --tail=40
```

**5. Forward the API to the Linux (VM) shell.** The LoadBalancer service may show `<pending>` in Minikube; that is not sufficient evidence of a failed deployment. Open a separate terminal on the VM:

```bash
minikube kubectl -- -n heart-disease port-forward \
  svc/heart-disease-api 18080:8000 --address 127.0.0.1
```

Keep the port-forward running. From another **Linux (VM)** terminal, check:

```bash
curl -f http://127.0.0.1:18080/health
curl -f http://127.0.0.1:18080/ready
curl -f http://127.0.0.1:18080/model-info
```

The API URL above is on the **VM**. To use a browser on a separate computer, configure an authorized SSH tunnel or another appropriate access method; it is not a public URL.

For recovery after the VM is replaced, see [Linux (VM) recovery guide](docs/prayogshala-recovery.md). The guide covers the persisted MLflow backend/artifacts and an optional saved Docker image. **A new VM also requires restarting MLflow, the bridge, Minikube, and port forwarding.**

## 11. API Endpoints and Prediction — Mac and Linux (VM)

| Endpoint      | Method | Purpose                                          |
| ------------- | ------ | ------------------------------------------------ |
| `/health`     | GET    | Service and model health                         |
| `/ready`      | GET    | Model readiness                                  |
| `/model-info` | GET    | Registered model details                         |
| `/predict`    | POST   | Binary prediction and positive-class probability |
| `/metrics`    | GET    | Request/error counts and latency                 |
| `/monitor`    | GET    | Monitoring dashboard                             |
| `/docs`       | GET    | Swagger UI                                       |

Set `API_BASE` to the address for your environment:

```bash
# Mac
API_BASE=http://127.0.0.1:8088

# Linux (VM): instead use the following after port-forwarding
# API_BASE=http://127.0.0.1:18080

curl -f "$API_BASE/health"
curl -f "$API_BASE/ready"
curl -f "$API_BASE/model-info"

curl -sS -X POST "$API_BASE/predict" \
  -H 'Content-Type: application/json' \
  -d '{"age":54,"sex":1,"cp":2,"trestbps":130,"chol":230,"fbs":0,"restecg":0,"thalach":160,"exang":0,"oldpeak":1.0,"slope":2,"ca":0,"thal":3}'
```

One verified **Mac** response was:

```json
{
  "prediction": 0,
  "probability": 0.21225669206996303,
  "model_version": "1",
  "model_alias": "champion"
}
```

The `probability` field is the predicted probability of **Class 1**, not necessarily the confidence in the predicted class. Values and registered model versions may differ across independently trained environments.

## 12. Monitoring and Logs

The FastAPI monitoring dashboard displays prediction requests, successful requests, HTTP 4xx/5xx errors, average latency and p95 latency. It refreshes every five seconds, uses **in-memory counters**, and does not retain patient inputs. Metrics reset when the API process restarts.

```bash
# Set API_BASE as shown in Section 11
curl -sS "$API_BASE/metrics" | python -m json.tool
```

View logs:

```bash
# Mac
kubectl -n heart-disease logs deployment/heart-disease-api --tail=25

# Linux (VM)
minikube kubectl -- -n heart-disease logs deployment/heart-disease-api --tail=25
```

### Measured latency baseline

The recorded benchmark used **200 sequential prediction requests following five warm-up requests**:

| Metric                 | Observed value |
| ---------------------- | -------------: |
| Successful predictions |        200/200 |
| Average latency        |       27.93 ms |
| p50 latency            |       22.75 ms |
| p95 latency            |       51.32 ms |
| p99 latency            |       56.14 ms |

A separate reliability test recorded 100 successful valid requests and 10 correctly rejected invalid requests. During a controlled MLflow outage, a running pod continued serving its already-loaded model, while a newly started isolated pod did not become ready during a 42-second observation. See `docs/monitoring_baseline.md` for the test conditions.

**The benchmark above is separate from the current `/monitor` dashboard readings**, which depend on requests handled since the API process last started.

## 13. GitHub Actions CI

The workflow at `.github/workflows/ci.yml` runs on pushes and pull requests targeting `main`:

1. **Python Tests:** Ruff linting and 22 Pytest tests.
2. **Model Training:** Baseline five-fold cross-validation and upload of the `model-training-results` artifact.
3. **Docker Build:** Image-build validation.

The three jobs are dependent. CI does **not** automatically deploy Kubernetes, publish a production image, or promote an MLflow model.

## 14. Troubleshooting — Mac and Linux (VM)

| Problem                                                 | What to check                                                                                               |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `.venv/bin/activate` not found                          | Confirm whether `.venv` is a Python venv or a Conda environment. Activate it using the appropriate command. |
| `kubectl` cannot reach `colima-mlops` (**Mac**)         | Run `colima list`; start `--profile mlops` and confirm the selected Kubernetes context.                     |
| `mlops` profile missing (**Mac**)                       | Create it with `colima start --profile mlops --kubernetes --cpu 4 --memory 6 --disk 20`.                    |
| Docker image missing (**Mac**)                          | Select Docker context `colima-mlops` and rebuild `heart-disease-api:0.2`.                                   |
| MLflow health check fails                               | Restart the server from the repository root. Confirm its database/artifacts and port 5050.                  |
| Champion model not found                                | Register an available MLflow model and set alias `champion` as in Section 8.4.                              |
| Pod fails to become ready                               | Inspect pod events/logs, MLflow reachability, the registered model and artifact paths.                      |
| `host.lima.internal` unavailable (**Mac**)              | Check the Colima profile and MLflow host binding/allowed hosts.                                             |
| `host.minikube.internal` or bridge fails (**Linux VM**) | Check Minikube networking, bridge IP, process log and port 5051.                                            |
| Minikube cannot find the image (**Linux VM**)           | Re-run `minikube image load heart-disease-api:0.2`.                                                         |
| API URL does not respond                                | Check Kubernetes Service, pod readiness, and the appropriate port-forward.                                  |
| Dashboard counters reset                                | In-memory monitoring is reset on API process restarts.                                                      |

## 15. Reports and Evidence

- `reports/figures/` — EDA and evaluation charts
- `reports/model_selection.md` — Model comparison and selection rationale
- `reports/mlflow_experiment_summary.md` — Experiment tracking summary
- `reports/test_metrics.json` — Held-out test metrics
- `docs/architecture.md` — System architecture
- `docs/monitoring_baseline.md` — Latency and reliability results
- `docs/screenshots/` — Captured evidence from the workflow
- `docs/prayogshala-recovery.md` — Linux (VM) restart/recovery procedure
- `k8s/heart-disease-api.yaml` — Mac Kubernetes manifest
- `k8s/heart-disease-api-prayogshala.yaml` — Linux (VM) Kubernetes manifest
- `submission/Heart Disease Prediction Final Report.docx` — Assignment report

## 16. Limitations

- The dataset is small and the reported metrics do not establish clinical validity.
- The original experiment runs and MLflow registry are not included in Git.
- Some registration utilities depend on run/model IDs created in a particular MLflow database.
- GitHub Actions validates the pipeline but does not deploy the serving environment.
- Mac and Linux (VM) use different Kubernetes and MLflow networking configurations.
- The API metrics are process-memory counters rather than a persistent monitoring database.
- The latency benchmark uses sequential traffic, not concurrent load testing.
- Serving access is configured for the specified Mac or Linux (VM) environment and is not an internet-facing deployment.
