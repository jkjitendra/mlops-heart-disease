"""Dataset loading and reproducible train/test splitting."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "heart_disease_prepared.csv"
)

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_data():
    """Load model features and binary target separately."""

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    if set(df.columns) != set(required_columns):
        raise ValueError("Unexpected dataset columns")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Missing target values detected")

    if not df[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("Target must contain only 0 and 1")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


def split_data(X, y):
    """Create a reproducible stratified 80/20 split."""

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
