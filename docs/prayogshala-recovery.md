# BITS Prayogshala — MLflow and Kubernetes Recovery Guide

This guide restores the Heart Disease Prediction MLOps demo after a
Prayogshala VM change.

## Persistent files

- Repository: `~/workspace/mlops-heart-disease`
- Python environment: `.venv/`
- MLflow database: `mlflow.db`
- MLflow artifacts: `mlartifacts/`
- Docker image backup:
  `~/workspace/mlops-recovery/heart-disease-api-0.2.tar.gz`

The MLflow database and artifacts are stored in the persistent EFS
workspace. Docker images and running Minikube containers may not survive
a VM replacement.

## 1. Start MLflow — Terminal 1

```bash
cd ~/workspace/mlops-heart-disease
source .venv/bin/activate

mlflow server \
  --backend-store-uri "sqlite:///$PWD/mlflow.db" \
  --artifacts-destination "$PWD/mlartifacts" \
  --host 127.0.0.1 \
  --port 5050 \
  --workers 1 \
  --allowed-hosts "localhost:*,127.0.0.1:*"
```

Keep Terminal 1 open. Verify that the server reports successful startup.

## 2. Restore Docker and Minikube — Terminal 2

```bash
cd ~/workspace/mlops-heart-disease
source .venv/bin/activate

gzip -dc ~/workspace/mlops-recovery/heart-disease-api-0.2.tar.gz \
  | docker load

minikube start --driver=docker --cpus=2 --memory=4096

minikube image load heart-disease-api:0.2

minikube kubectl -- get nodes
```

If the image archive is unavailable, rebuild using
`docker build -t heart-disease-api:0.2 .`.

## 3. Start the private MLflow bridge — Terminal 2

Identify Minikube's host bridge address:

```bash
export MLFLOW_BRIDGE_HOST="$(
  minikube ssh -- getent hosts host.minikube.internal |
  awk 'NR==1 {print $1}'
)"

echo "Bridge address: $MLFLOW_BRIDGE_HOST"
test -n "$MLFLOW_BRIDGE_HOST"
```

Start the private bridge:

```bash
nohup env MLFLOW_BRIDGE_HOST="$MLFLOW_BRIDGE_HOST" \
  .venv/bin/python -u scripts/mlflow_vm_bridge.py \
  > ~/workspace/mlops-recovery/mlflow_bridge.log \
  2>&1 < /dev/null &

curl --fail \
  "http://$MLFLOW_BRIDGE_HOST:5051/health"
```

Expected response: `OK`.

The bridge is for a private academic demonstration environment,
not a production authenticated MLflow gateway.

## 4. Deploy the Kubernetes application

Generate a VM-specific runtime manifest using the current bridge IP:

```bash
sed \
  "s|http://192.168.49.1:5051|http://${MLFLOW_BRIDGE_HOST}:5051|" \
  k8s/heart-disease-api-prayogshala.yaml \
  > ~/workspace/mlops-recovery/heart-disease-api-runtime.yaml

minikube kubectl -- apply \
  -f ~/workspace/mlops-recovery/heart-disease-api-runtime.yaml

minikube kubectl -- rollout status \
  deployment/heart-disease-api \
  -n heart-disease \
  --timeout=240s
```

Verify the deployed pod and model:

```bash
minikube kubectl -- get pods,services -n heart-disease

minikube kubectl -- logs \
  deployment/heart-disease-api \
  -n heart-disease --tail=30
```

Expected: a ready pod and successful champion model loading.

## 5. Access the Kubernetes FastAPI service

In Terminal 2, run:

```bash
minikube kubectl -- -n heart-disease port-forward \
  svc/heart-disease-api \
  18080:8000 \
  --address 127.0.0.1
```

Keep this terminal running. In another VM terminal:

```bash
curl --fail http://127.0.0.1:18080/health
curl --fail http://127.0.0.1:18080/ready
curl --fail http://127.0.0.1:18080/model-info
curl --fail http://127.0.0.1:18080/metrics
```

The monitoring dashboard is available at:

`http://127.0.0.1:18080/monitor`

These URLs are local to the VM. The Minikube LoadBalancer may show
`EXTERNAL-IP: <pending>`; use port forwarding for local verification.

## Important notes

- Do not recreate the registered model if the persistent MLflow
  database and artifacts already exist.
- Confirm the current Minikube bridge IP after every VM replacement.
- The existing Mac/Colima manifest remains unchanged.
- A successful Kubernetes deployment does not mean that its running
  state will persist when Prayogshala replaces the VM.
- This is an academic demonstration, not a clinical prediction service.
