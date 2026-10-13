# Model Selection and Hyperparameter Tuning

## Dataset and Evaluation Strategy

Dataset: UCI Heart Disease (Processed Cleveland)

- Total records: 303
- Input features: 13
- Binary target: heart_disease
- Training records: 242
- Held-out testing records: 61
- Train/test split: 80/20 stratified
- Random seed: 42
- Cross-validation: 5-fold StratifiedKFold

All preprocessing operations are performed inside
scikit-learn Pipelines to prevent data leakage.

The held-out test set was not used during model
comparison or hyperparameter tuning.

## Baseline Models

Two classifiers were evaluated:

1. Logistic Regression
2. Random Forest

## Cross-Validation Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression Baseline | 0.8471 | 0.8768 | 0.7834 | 0.8245 | 0.9025 |
| Logistic Regression Tuned | 0.8304 | 0.8232 | 0.8099 | 0.8132 | 0.9028 |
| Random Forest Baseline | 0.8139 | 0.8129 | 0.7739 | 0.7900 | 0.8923 |
| Random Forest Tuned | 0.8056 | 0.7965 | 0.7826 | 0.7854 | 0.9044 |

All reported values are mean cross-validation scores
calculated using training data only.

## Hyperparameter Tuning

### Logistic Regression

Evaluated:
- C: 0.1, 1.0, 10.0
- class_weight: None, balanced

Best parameters by ROC-AUC:
- C: 0.1
- class_weight: balanced

Best CV ROC-AUC: 0.9028

### Random Forest

Evaluated:
- max_depth: None, 5
- min_samples_leaf: 1, 3
- class_weight: None, balanced

Fixed parameters:
- n_estimators: 200
- random_state: 42

Best parameters by ROC-AUC:
- max_depth: 5
- min_samples_leaf: 3
- class_weight: balanced

Best CV ROC-AUC: 0.9044
ROC-AUC standard deviation: 0.0260

## Model Selection Decision

Selection criterion: Highest mean cross-validation ROC-AUC.

Selected candidate: Tuned Random Forest.

The tuned Random Forest achieved the highest mean
cross-validation ROC-AUC of 0.9044.

However, its advantage over tuned Logistic Regression
was only 0.0016. This small difference should not be
interpreted as evidence of statistically significant
superiority.

Logistic Regression also achieved better accuracy,
recall, and F1-score. These trade-offs are acknowledged.

The model selection decision is fixed before evaluating
the held-out test dataset.

## Next Step

Refit the selected Random Forest pipeline using the
242 training records and evaluate it once using the
61 held-out test records.

Final test-set metrics will be documented separately.
