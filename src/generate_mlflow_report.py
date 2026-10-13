"""Generate comparison chart and report from MLflow export."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "reports/mlflow_model_comparison.csv"

FIGURE_PATH = (
    ROOT / "reports/figures/mlflow_model_comparison.png"
)

REPORT_PATH = ROOT / "reports/mlflow_experiment_summary.md"

MODEL_ORDER = [
    "Logistic Regression",
    "Random Forest",
    "SVM (RBF)",
    "Gradient Boosting",
]

METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
]


def validate_data(df):
    """Check experiment completeness and metric validity."""

    required_columns = {
        "model",
        "configuration",
        "run_name",
        "run_id",
        *METRICS,
    }

    if not required_columns.issubset(df.columns):
        raise ValueError("Missing required experiment columns")

    if len(df) != 8:
        raise ValueError("Expected exactly eight experiment runs")

    if df.duplicated(["model", "configuration"]).any():
        raise ValueError("Duplicate model configurations detected")

    expected_pairs = {
        (model, config)
        for model in MODEL_ORDER
        for config in ["Baseline", "Tuned"]
    }

    actual_pairs = set(
        zip(df["model"], df["configuration"])
    )

    if actual_pairs != expected_pairs:
        raise ValueError("Unexpected model configurations")

    if df["run_id"].isna().any() or not df["run_id"].is_unique:
        raise ValueError("Invalid or duplicate MLflow Run IDs")

    if not df[METRICS].notna().all().all():
        raise ValueError("Missing evaluation metrics")

    if not df[METRICS].apply(
        lambda column: column.between(0, 1).all()
    ).all():
        raise ValueError("Metrics must be between 0 and 1")


def create_chart(df):
    """Plot baseline and tuned ROC-AUC values."""

    comparison = df.pivot(
        index="model",
        columns="configuration",
        values="roc_auc",
    ).loc[MODEL_ORDER]

    x = np.arange(len(MODEL_ORDER))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    baseline_bars = ax.bar(
        x - width / 2,
        comparison["Baseline"],
        width,
        label="Baseline",
        color="#3978B7",
    )

    tuned_bars = ax.bar(
        x + width / 2,
        comparison["Tuned"],
        width,
        label="Tuned",
        color="#299486",
    )

    for bars in [baseline_bars, tuned_bars]:
        ax.bar_label(
            bars,
            labels=[
                f"{bar.get_height():.2%}"
                for bar in bars
            ],
            padding=3,
            fontsize=9,
        )

    ax.set_title(
        "Heart Disease Classification - MLflow Comparison"
    )
    ax.set_xlabel("Classification Model")
    ax.set_ylabel("Cross-Validation ROC-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower center", ncol=2, frameon=True)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    FIGURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        FIGURE_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def create_report(df):
    """Write a reproducible Markdown experiment summary."""

    lines = [
        "# MLflow Experiment Comparison",
        "",
        "## Experiment Overview",
        "",
        "- Dataset: UCI Heart Disease (Processed Cleveland)",
        "- Total records: 303",
        "- Training records: 242",
        "- Held-out testing records: 61",
        "- Cross-validation: 5-fold StratifiedKFold",
        "- Random state: 42",
        "- Primary selection metric: ROC-AUC",
        "- Model families: 4",
        "- Reported model configurations: 8",
        "",
        "All comparison scores come from training-set "
        "cross-validation. The held-out test set is "
        "excluded from this expanded comparison.",
        "",
        "## Model Comparison",
        "",
        "| Model | Configuration | Accuracy | Precision "
        "| Recall | F1 | ROC-AUC |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for model in MODEL_ORDER:
        for config in ["Baseline", "Tuned"]:
            row = df[
                (df["model"] == model)
                & (df["configuration"] == config)
            ].iloc[0]

            values = " | ".join(
                f"{row[metric]:.4f}"
                for metric in METRICS
            )

            lines.append(
                f"| {model} | {config} | {values} |"
            )

    lines.extend([
        "",
        "## Hyperparameter Tuning Improvements",
        "",
        "| Model | Baseline ROC-AUC | Tuned ROC-AUC "
        "| Difference (pp) |",
        "|---|---:|---:|---:|",
    ])

    for model in MODEL_ORDER:
        subset = df[df["model"] == model]

        baseline = subset[
            subset["configuration"] == "Baseline"
        ]["roc_auc"].iloc[0]

        tuned = subset[
            subset["configuration"] == "Tuned"
        ]["roc_auc"].iloc[0]

        difference_pp = (tuned - baseline) * 100

        lines.append(
            f"| {model} | {baseline:.4f} "
            f"| {tuned:.4f} | {difference_pp:+.2f} |"
        )

    tuned_df = df[df["configuration"] == "Tuned"]
    best = tuned_df.loc[tuned_df["roc_auc"].idxmax()]

    lines.extend([
        "",
        "## Highest Tuned Cross-Validation ROC-AUC",
        "",
        f"- Model: {best['model']}",
        f"- Mean ROC-AUC: {best['roc_auc']:.4f}",
        f"- MLflow Run ID: `{best['run_id']}`",
        "",
        "This is the highest observed score, not proof "
        "of statistically significant superiority.",
        "",
        "## Evaluation Methodology and Limitations",
        "",
        "All four model families were compared using "
        "the same training split and cross-validation "
        "procedure.",
        "",
        "Hyperparameter selection can make the winning "
        "cross-validation score optimistic. The dataset "
        "is small, and performance differences should "
        "be interpreted cautiously.",
        "",
        "The tuned Random Forest was originally selected "
        "and evaluated on the 61-record held-out test "
        "set before the comparison was expanded to four "
        "model families.",
        "",
        "That earlier held-out test result is documented "
        "separately in `reports/test_metrics.json`. "
        "It is not used to rank the expanded model set.",
        "",
        "This academic classifier has not been validated "
        "for clinical decision-making.",
        "",
        "## Reproducibility",
        "",
        "The source metrics were exported from MLflow "
        "using `src/export_mlflow.py`.",
        "",
        "The comparison CSV contains the MLflow Run IDs "
        "needed to trace the recorded results.",
        "",
        "To regenerate this report and its figure:",
        "",
        "```bash",
        "python -m src.export_mlflow",
        "python -m src.generate_mlflow_report",
        "```",
        "",
    ])

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    df = pd.read_csv(INPUT_PATH)

    validate_data(df)

    create_chart(df)
    create_report(df)

    print("=== MLFLOW REPORT GENERATION ===")
    print(f"Validated experiments: {len(df)}")
    print(f"Generated: {FIGURE_PATH.relative_to(ROOT)}")
    print(f"Generated: {REPORT_PATH.relative_to(ROOT)}")
    print("MLflow comparison report completed!")


if __name__ == "__main__":
    main()
