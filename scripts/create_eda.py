"""Generate exploratory visualizations for the Cleveland dataset."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/raw/processed.cleveland.data"
OUTPUT_DIR = ROOT / "reports/figures"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "num",
]

NUMERIC_FEATURES = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

FEATURE_LABELS = {
    "age": "Age (years)",
    "trestbps": "Resting Blood Pressure",
    "chol": "Serum Cholesterol",
    "thalach": "Maximum Heart Rate",
    "oldpeak": "ST Depression",
}

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = pd.read_csv(
        DATA_PATH,
        header=None,
        names=COLUMNS,
        na_values=["?"],
    )

    df["heart_disease"] = (df["num"] > 0).astype(int)

    # Chart 1: Binary class distribution
    counts = df["heart_disease"].value_counts().reindex(
        [0, 1], fill_value=0
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        ["No Heart Disease", "Heart Disease Present"],
        counts.values,
        color=["#308D85", "#D68A49"],
        width=0.55,
    )

    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count + 3,
            f"{count} ({count / len(df):.1%})",
            ha="center",
        )

    ax.set_title("Heart Disease Class Distribution")
    ax.set_ylabel("Number of Records")
    ax.set_ylim(0, max(counts) * 1.2)

    fig.savefig(
        OUTPUT_DIR / "class_balance.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Chart 2: Numerical feature distributions
    fig, axes = plt.subplots(
        2, 3, figsize=(13, 7), layout="constrained"
    )

    for ax, feature in zip(axes.flat, NUMERIC_FEATURES):
        sns.histplot(
            df[feature],
            bins=18,
            ax=ax,
            color="#3978B7",
        )
        ax.set_xlabel(FEATURE_LABELS[feature])
        ax.set_ylabel("Frequency")

    axes.flat[-1].axis("off")
    fig.suptitle("Numerical Feature Distributions", fontsize=16)

    fig.savefig(
        OUTPUT_DIR / "feature_histograms.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Chart 3: Numerical correlation heatmap
    correlation_data = df[
        NUMERIC_FEATURES + ["heart_disease"]
    ]

    fig, ax = plt.subplots(figsize=(9, 7))

    sns.heatmap(
        correlation_data.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
    )

    ax.set_title("Numerical Feature Correlation Heatmap")

    fig.savefig(
        OUTPUT_DIR / "correlation_heatmap.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("EDA visualizations generated successfully!")

    for path in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"Created: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
