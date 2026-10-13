# MLflow Experiment Comparison

## Experiment Overview

- Dataset: UCI Heart Disease (Processed Cleveland)
- Total records: 303
- Training records: 242
- Held-out testing records: 61
- Cross-validation: 5-fold StratifiedKFold
- Random state: 42
- Primary selection metric: ROC-AUC
- Model families: 4
- Reported model configurations: 8

All comparison scores come from training-set cross-validation. The held-out test set is excluded from this expanded comparison.

## Model Comparison

| Model | Configuration | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | Baseline | 0.8471 | 0.8768 | 0.7834 | 0.8245 | 0.9025 |
| Logistic Regression | Tuned | 0.8304 | 0.8232 | 0.8099 | 0.8132 | 0.9028 |
| Random Forest | Baseline | 0.8139 | 0.8129 | 0.7739 | 0.7900 | 0.8923 |
| Random Forest | Tuned | 0.8056 | 0.7965 | 0.7826 | 0.7854 | 0.9044 |
| SVM (RBF) | Baseline | 0.8139 | 0.8303 | 0.7648 | 0.7893 | 0.8884 |
| SVM (RBF) | Tuned | 0.8139 | 0.8303 | 0.7648 | 0.7893 | 0.8884 |
| Gradient Boosting | Baseline | 0.7975 | 0.7724 | 0.7925 | 0.7809 | 0.8563 |
| Gradient Boosting | Tuned | 0.8469 | 0.8777 | 0.7830 | 0.8234 | 0.8958 |

## Hyperparameter Tuning Improvements

| Model | Baseline ROC-AUC | Tuned ROC-AUC | Difference (pp) |
|---|---:|---:|---:|
| Logistic Regression | 0.9025 | 0.9028 | +0.03 |
| Random Forest | 0.8923 | 0.9044 | +1.21 |
| SVM (RBF) | 0.8884 | 0.8884 | +0.00 |
| Gradient Boosting | 0.8563 | 0.8958 | +3.94 |

## Highest Tuned Cross-Validation ROC-AUC

- Model: Random Forest
- Mean ROC-AUC: 0.9044
- MLflow Run ID: `3bf028ad8b7b45e89766e325703d38f3`

This is the highest observed score, not proof of statistically significant superiority.

## Evaluation Methodology and Limitations

All four model families were compared using the same training split and cross-validation procedure.

Hyperparameter selection can make the winning cross-validation score optimistic. The dataset is small, and performance differences should be interpreted cautiously.

The tuned Random Forest was originally selected and evaluated on the 61-record held-out test set before the comparison was expanded to four model families.

That earlier held-out test result is documented separately in `reports/test_metrics.json`. It is not used to rank the expanded model set.

This academic classifier has not been validated for clinical decision-making.

## Reproducibility

The source metrics were exported from MLflow using `src/export_mlflow.py`.

The comparison CSV contains the MLflow Run IDs needed to trace the recorded results.

To regenerate this report and its figure:

```bash
python -m src.export_mlflow
python -m src.generate_mlflow_report
```
