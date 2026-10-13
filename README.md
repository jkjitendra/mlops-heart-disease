# Heart Disease Prediction — End-to-End MLOps

## Project Overview

This project is developed as part of the BITS Pilani
MLOps Experimental Learning Assignment 1.

The objective is to build a reproducible machine learning
pipeline for heart disease classification and demonstrate
MLOps practices such as experiment tracking, automated
testing, CI/CD, containerization, deployment, and monitoring.

## Dataset

Source: UCI Machine Learning Repository

Dataset: Heart Disease (Cleveland)

URL: https://archive.ics.uci.edu/dataset/45/heart+disease

The classification task uses patient health attributes to
predict the presence or absence of heart disease.

The original target values will be converted into binary
labels:
- 0: Absence of heart disease
- 1-4: Presence of heart disease (converted to 1)

The application is intended for educational purposes only,
not for clinical diagnosis or medical decision-making.

## Planned Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- MLflow
- FastAPI
- Pytest and Ruff
- Git and GitHub
- GitHub Actions
- Docker
- Kubernetes (local deployment)
- API logging and monitoring

## Planned MLOps Workflow

1. Dataset acquisition and exploratory data analysis
2. Feature engineering and preprocessing
3. Model training, tuning, and evaluation
4. Experiment tracking using MLflow
5. Model packaging and reproducibility
6. Automated testing and CI/CD
7. Docker-based API serving
8. Local Kubernetes deployment
9. Monitoring and logging
10. Documentation and reporting

## Development Environment

- Hardware: Apple Silicon Mac (M1, 16 GB RAM)
- Storage: External SSD
- Project filesystem: APFS sparse disk image
- Workspace: /Volumes/MLOpsWorkspace

### Opening the Local Workspace

The development machine has a local startup script:

```bash
source ~/open-mlops.sh
```

This mounts the APFS workspace and opens the project
directory. The script is specific to the development
machine and is not a project dependency.

### Closing the Local Workspace

After stopping project services:

```bash
source ~/close-mlops.sh
```

## Project Status

Status: Initial repository and environment setup.

Implementation, evaluation results, API usage,
deployment instructions, and monitoring details
will be documented as development progresses.

## Repository

https://github.com/jkjitendra/mlops-heart-disease

## License and Usage

This project is developed for academic learning and
demonstration purposes.
