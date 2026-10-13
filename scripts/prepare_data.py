"""Prepare the UCI Cleveland Heart Disease dataset for modelling."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data/raw/processed.cleveland.data"
OUTPUT_PATH = ROOT / "data/processed/heart_disease_prepared.csv"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "num",
]


def prepare_data():
    # Load raw data and recognize missing-value placeholders.
    df = pd.read_csv(
        RAW_PATH,
        header=None,
        names=COLUMNS,
        na_values=["?"],
    )

    # Validate the expected dataset structure.
    if df.shape != (303, 14):
        raise ValueError(
            f"Unexpected dataset shape: {df.shape}"
        )

    if df.duplicated().any():
        raise ValueError("Duplicate records detected")

    # The original target must contain values from 0 to 4.
    if not df["num"].isin([0, 1, 2, 3, 4]).all():
        raise ValueError("Invalid or missing target values")

    # Convert the original target into binary classification.
    df["heart_disease"] = (df["num"] > 0).astype(int)

    # Remove the original target to prevent target leakage.
    df = df.drop(columns=["num"])

    return df


def main():
    df = prepare_data()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Keep missing features empty for train-fitted imputation later.
    df.to_csv(OUTPUT_PATH, index=False)

    print("Dataset preparation successful!")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Total records: {len(df)}")
    print(f"Input features: {df.shape[1] - 1}")
    print(f"Missing values: {df.isna().sum().sum()}")
    print(f"Rows with missing values: {df.isna().any(axis=1).sum()}")

    print("\nBinary target distribution:")
    print(df["heart_disease"].value_counts().sort_index())

    print("\nOutput columns:")
    print(list(df.columns))


if __name__ == "__main__":
    main()
