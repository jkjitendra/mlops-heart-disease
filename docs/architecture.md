# Heart Disease MLOps Architecture

This diagram describes the implemented local MLOps system.
GitHub Actions performs CI validation; Kubernetes deployment
is performed locally and is not an automatic CI release.

```mermaid
flowchart TB

    subgraph DEVELOPMENT["1. Data and Model Development"]
        A["UCI Cleveland Dataset<br/>303 records"]
        B["Data Preparation and EDA"]
        C["Prepared Dataset<br/>13 features and binary target"]
        D["Stratified 80/20 Train-Test Split"]
        E["Training and Tuning<br/>5-fold cross-validation"]
        F["MLflow Experiment Tracking<br/>Parameters, metrics and artifacts"]
        G["Tuned Random Forest<br/>Selected champion"]
        H["Held-out Test Evaluation<br/>61 records"]
        I["MLflow Model Registry<br/>Champion alias, version 1"]

        A --> B --> C --> D --> E
        E --> F
        E --> G
        G --> H
        G --> I
    end

    subgraph CI["2. GitHub Actions Continuous Integration"]
        J["GitHub Repository"]
        K["Ruff Linting"]
        L["Pytest<br/>22 automated tests"]
        M["Baseline Model Training<br/>Cross-validation"]
        N["Training Log Artifact"]
        O["Docker Image Build Validation"]

        J --> K --> L --> M
        M --> N
        M --> O
    end

    subgraph SERVING["3. Local Docker and Kubernetes Serving"]
        P["Docker Image<br/>heart-disease-api:0.2"]
        Q["Colima and K3s<br/>Kubernetes Deployment"]
        R["FastAPI<br/>Model loaded at startup"]
        S["LoadBalancer<br/>127.0.0.1:8088"]
        T["Prediction API<br/>/predict"]
        U["Monitoring Dashboard<br/>/monitor and /metrics"]
        V["Kubernetes API Logs"]

        P --> Q --> R --> S
        S --> T
        S --> U
        R --> V
    end

    J -. "Source code and Dockerfile" .-> P
    I -. "Retrieve champion on startup" .-> R

```

## Architecture Notes

- The dataset is prepared from the UCI Cleveland source.
- Scikit-learn Pipelines include preprocessing and classifiers.
- Cross-validation and tuning use training data only.
- The selected tuned Random Forest is registered in MLflow.
- The FastAPI application loads the registered model at startup.
- The serving container runs in local Kubernetes using Colima and K3s.
- The Kubernetes LoadBalancer is reachable from the host on port 8088.
- API metrics are held in memory and reset when the process restarts.
- GitHub Actions runs linting, tests, baseline training and Docker build validation.
- CI does not automatically publish the image or deploy to Kubernetes.
- The application is an academic demonstration, not a clinical system.
