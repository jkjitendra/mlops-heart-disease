"""Inspect the original UCI Cleveland Heart Disease dataset."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data/raw/processed.cleveland.data"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "num",
]


def main():
    # Load the original dataset without modifying the source file.
    df = pd.read_csv(
        DATA_PATH,
        header=None,
        names=COLUMNS,
        na_values=["?"],
    )

    print("=== DATASET OVERVIEW ===")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\n=== FIRST 5 RECORDS ===")
    print(df.head().to_string(index=False))

    print("\n=== COLUMN DATA TYPES ===")
    print(df.dtypes.to_string())

    print("\n=== MISSING VALUES PER COLUMN ===")
    print(df.isna().sum().to_string())
    print(f"Rows with missing values: {df.isna().any(axis=1).sum()}")

    print("\n=== DUPLICATE RECORDS ===")
    print(f"Exact duplicates: {df.duplicated().sum()}")

    print("\n=== ORIGINAL TARGET DISTRIBUTION ===")
    print(df["num"].value_counts().sort_index().to_string())

    # Temporary derived target for inspection only.
    binary_target = (df["num"] > 0).astype(int)

    print("\n=== BINARY TARGET DISTRIBUTION ===")
    print(binary_target.value_counts().sort_index().to_string())

    print("\n=== NUMERICAL SUMMARY ===")
    print(df.describe().round(2).to_string())


if __name__ == "__main__":
    main()
