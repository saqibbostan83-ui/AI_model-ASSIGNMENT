"""
Exploratory Data Analysis plots. Every function draws one figure; figures are
also saved to reports/figures/ when the module is run as a script.

Usage:
    python src/eda.py           # regenerate all EDA figures
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

import matplotlib
if "ipykernel" not in sys.modules:  # keep inline plots working in Jupyter
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.preprocess import get_dataset

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})

PALETTE = {"NO": "#4c72b0", ">30": "#dd8452", "<30": "#c44e52"}


def plot_target_distribution(df: pd.DataFrame) -> plt.Figure:
    """Distribution of the 3-level readmitted column + binary target."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    order = ["NO", ">30", "<30"]
    counts = df["readmitted"].value_counts().reindex(order)
    bars = axes[0].bar(order, counts.values, color=[PALETTE[o] for o in order])
    axes[0].set_title("Readmission class distribution (original target)")
    axes[0].set_ylabel("encounters")
    for b, v in zip(bars, counts.values):
        axes[0].annotate(f"{v:,}\n({v/len(df):.1%})", (b.get_x() + b.get_width()/2, v),
                         ha="center", va="bottom", fontsize=9)
    axes[0].margins(y=0.25)

    binary = (df["readmitted"] == "<30").map({1: "Readmitted <30d", 0: "Not <30d"}).value_counts()
    axes[1].bar(binary.index, binary.values, color=["#4c72b0", "#c44e52"])
    axes[1].set_title("Binary target: readmitted within 30 days")
    for i, v in enumerate(binary.values):
        axes[1].annotate(f"{v:,}\n({v/len(df):.1%})", (i, v), ha="center", va="bottom", fontsize=9)
    axes[1].margins(y=0.25)
    return fig


def plot_missing_values(df: pd.DataFrame, top: int = 15) -> plt.Figure:
    """Columns with the most missing values (the dataset codes missing as '?')."""
    miss = df.isna().mean().mul(100).sort_values(ascending=False)
    miss = miss[miss > 0].head(top)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(miss.index[::-1], miss.values[::-1], color="#dd8452")
    ax.set_xlabel("% of rows missing")
    ax.set_title(f"Columns with missing values (top {len(miss)}) — dropped or imputed in preprocessing")
    for i, v in enumerate(miss.values[::-1]):
        ax.annotate(f"{v:.1f}%", (v, i), va="center", fontsize=9)
    ax.set_xlim(0, 105)
    return fig


def plot_numeric_distributions(df: pd.DataFrame) -> plt.Figure:
    cols = ["age", "time_in_hospital", "num_lab_procedures", "num_procedures",
            "num_medications", "number_diagnoses"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.flat, cols):
        ax.hist(df[col].dropna(), bins=30, color="#4c72b0", edgecolor="white", linewidth=.3)
        ax.set_title(col, fontsize=10)
        ax.set_ylabel("count", fontsize=8)
    fig.suptitle("Distributions of key numeric features")
    fig.tight_layout()
    return fig


def plot_readmission_by_key_features(df: pd.DataFrame) -> plt.Figure:
    """Early-readmission rate across age, length of stay and prior utilisation."""
    df = df.copy()
    df["readmit30"] = (df["readmitted"] == "<30").astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    g = df.groupby("age")["readmit30"].mean()
    axes[0].plot(g.index, g.values * 100, marker="o", color="#c44e52")
    axes[0].set_xlabel("age (midpoint)"); axes[0].set_ylabel("% readmitted <30d")
    axes[0].set_title("Readmission rate vs age")

    g = df.groupby("time_in_hospital")["readmit30"].mean()
    axes[1].plot(g.index, g.values * 100, marker="o", color="#4c72b0")
    axes[1].set_xlabel("days in hospital"); axes[1].set_ylabel("% readmitted <30d")
    axes[1].set_title("Readmission rate vs length of stay")

    g = df.groupby(pd.cut(df["service_utilization"], [-1, 0, 2, 5, 10, 200],
                          labels=["0", "1-2", "3-5", "6-10", ">10"]), observed=True)["readmit30"].mean()
    axes[2].bar(g.index.astype(str), g.values * 100, color="#55a868")
    axes[2].set_xlabel("outpatient+emergency+inpatient visits (prior year)")
    axes[2].set_ylabel("% readmitted <30d")
    axes[2].set_title("Readmission rate vs prior utilisation")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    cols = ["age", "time_in_hospital", "num_lab_procedures", "num_procedures",
            "num_medications", "number_outpatient", "number_emergency",
            "number_inpatient", "number_diagnoses", "service_utilization"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0,
                square=True, cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title("Correlation between numeric features")
    return fig


def plot_categorical_target(df: pd.DataFrame) -> plt.Figure:
    """Readmission rate by race, gender, discharge disposition and diagnosis group."""
    df = df.copy()
    df["readmit30"] = (df["readmitted"] == "<30").astype(int)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    order = df.groupby("race")["readmit30"].mean().sort_values().index
    sns.barplot(data=df, x="readmit30", y="race", order=order, ax=axes[0, 0], color="#4c72b0")
    axes[0, 0].set_xlabel("% readmitted <30d"); axes[0, 0].set_title("By race")

    order = df.groupby("discharge_disposition")["readmit30"].mean().sort_values().index
    sns.barplot(data=df, x="readmit30", y="discharge_disposition", order=order, ax=axes[0, 1], color="#dd8452")
    axes[0, 1].set_xlabel("% readmitted <30d"); axes[0, 1].set_title("By discharge disposition")

    order = df.groupby("primary_diag")["readmit30"].mean().sort_values().index
    sns.barplot(data=df, x="readmit30", y="primary_diag", order=order, ax=axes[1, 0], color="#55a868")
    axes[1, 0].set_xlabel("% readmitted <30d"); axes[1, 0].set_title("By primary diagnosis group (ICD-9)")

    order = df.groupby("admission_type")["readmit30"].mean().sort_values().index
    sns.barplot(data=df, x="readmit30", y="admission_type", order=order, ax=axes[1, 1], color="#c44e52")
    axes[1, 1].set_xlabel("% readmitted <30d"); axes[1, 1].set_title("By admission type")
    fig.tight_layout()
    return fig


def plot_medication_usage(df: pd.DataFrame) -> plt.Figure:
    """Usage frequency of the most common diabetes medications."""
    meds = ["insulin", "metformin", "glipizide", "glyburide", "pioglitazone",
            "rosiglitazone", "glimepiride", "repaglinide"]
    usage = {m: (df[m] != "No").mean() * 100 for m in meds}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(list(usage.keys())[::-1], list(usage.values())[::-1], color="#8172b3")
    ax.set_xlabel("% of patients with an active prescription")
    ax.set_title("Most frequently prescribed diabetes medications")
    for i, v in enumerate(list(usage.values())[::-1]):
        ax.annotate(f"{v:.1f}%", (v, i), va="center", fontsize=9)
    ax.set_xlim(0, max(usage.values()) * 1.2)
    return fig


ALL_PLOTS = {
    "01_target_distribution.png": lambda d: plot_target_distribution(d),
    "02_missing_values.png": lambda d: plot_missing_values(d),
    "03_numeric_distributions.png": lambda d: plot_numeric_distributions(d),
    "04_readmission_by_key_features.png": lambda d: plot_readmission_by_key_features(d),
    "05_correlation_heatmap.png": lambda d: plot_correlation_heatmap(d),
    "06_categorical_target_rates.png": lambda d: plot_categorical_target(d),
    "07_medication_usage.png": lambda d: plot_medication_usage(d),
}


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = get_dataset()
    df = data["clean"]
    for fname, fn in ALL_PLOTS.items():
        fig = fn(df)
        fig.savefig(FIG_DIR / fname)
        plt.close(fig)
        print(f"saved {fname}")


if __name__ == "__main__":
    main()
