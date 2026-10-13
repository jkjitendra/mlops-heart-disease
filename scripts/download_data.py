"""Download and validate the UCI Cleveland Heart Disease dataset."""

import csv
import hashlib
import io
from pathlib import Path
from urllib.request import urlopen

DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/heart-disease/"
    "processed.cleveland.data"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "processed.cleveland.data"

EXPECTED_ROWS = 303
EXPECTED_COLUMNS = 14


def main():
    if OUTPUT_PATH.exists():
        print("Dataset already exists. Reusing local copy.")
        data = OUTPUT_PATH.read_bytes()
    else:
        print("Downloading dataset from UCI...")

        with urlopen(DATA_URL, timeout=30) as response:
            data = response.read()

    # Validate the raw dataset before saving.
    text = data.decode("utf-8")
    records = list(csv.reader(io.StringIO(text)))

    if len(records) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} rows, found {len(records)}"
        )

    if any(len(row) != EXPECTED_COLUMNS for row in records):
        raise ValueError("Unexpected number of columns in dataset")

    # Save the original dataset without changing its values.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not OUTPUT_PATH.exists():
        OUTPUT_PATH.write_bytes(data)

    checksum = hashlib.sha256(data).hexdigest()

    print("Dataset validation successful!")
    print(f"Saved at: {OUTPUT_PATH}")
    print(f"Total records: {len(records)}")
    print(f"Total columns: {EXPECTED_COLUMNS}")
    print(f"SHA256: {checksum}")


if __name__ == "__main__":
    main()
