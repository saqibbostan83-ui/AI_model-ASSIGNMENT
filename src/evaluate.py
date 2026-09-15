"""
Evaluation figures & tables for the trained models.

Can be used two ways:
  * imported  -> plot functions take an in-memory results dict
                 (as produced by src.train.train_all_models)
  * CLI       -> python src/evaluate.py
                 regenerates figures from the saved artifacts in models/

Outputs (reports/figures/):
  08_roc_curves.png, 09_pr_curves.png, 10_confusion_matrices.png,
  11_feature_importance.png, 12_keras_training_history.png
  and reports/model_comparison.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

import matplotlib
if "ipykernel" not in sys.modules:  # keep inline plots working in Jupyter
    matplotlib.use("Agg")
matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, precision_recall_curve

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
FIG_DIR = REPO_ROOT / "reports" / "figures"

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})

MODEL_LABELS = {
    "logreg": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "keras_mlp": "Keras MLP (neural net)",
    "logreg_smote": "LogReg + SMOTE",
}
MODEL_COLORS = {
    "logreg": "#8c8c8c",
    "random_forest": "#55a868",
    "xgboost": "#c44e52",
    "keras_mlp": "#4c72b0",
    "logreg_smote": "#937860",
}


def plot_roc_curves(y_test, probas: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    from sklearn.metrics import roc_auc_score
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, color=MODEL_COLORS.get(name, None), lw=2,
                label=f"{MODEL_LABELS.get(name, name)} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — 30-day readmission prediction")
    ax.legend(loc="lower right", fontsize=9)
    return fig


def plot_pr_curves(y_test, probas: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    from sklearn.metrics import average_precision_score
    pos_rate = y_test.mean()
    for name, proba in probas.items():
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        ax.plot(rec, prec, color=MODEL_COLORS.get(name, None), lw=2,
                label=f"{MODEL_LABELS.get(name, name)} (AP = {ap:.3f})")
    ax.axhline(pos_rate, ls="--", color="grey", lw=1,
               label=f"baseline (positives = {pos_rate:.1%})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves (imbalanced target)")
    ax.legend(loc="upper right", fontsize=9)
    return fig


def plot_confusion_matrices(y_test, probas: dict, normalize: bool = True) -> plt.Figure:
    names = list(probas.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(3.1 * len(names), 3.4))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        cm = confusion_matrix(y_test, (probas[name] >= 0.5).astype(int))
        if normalize:
            cm = cm / cm.sum(axis=1, keepdims=True)
        ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"]).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(MODEL_LABELS.get(name, name), fontsize=10)
        ax.set_xlabel("Predicted <30d", fontsize=8)
        ax.set_ylabel("True <30d", fontsize=8)
    fig.suptitle("Confusion matrices at the 0.5 threshold (row-normalised)")
    fig.tight_layout()
    return fig


def plot_feature_importance(pipeline, top: int = 20) -> plt.Figure:
    """Top features of a fitted sklearn pipeline (tree importances or |coef|)."""
    preproc = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]
    names = preproc.get_feature_names_out()

    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    else:
        imp = np.abs(clf.coef_[0])
    order = np.argsort(imp)[::-1][:top]
    labels = [str(names[i]) for i in order][::-1]
    values = imp[order][::-1]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(labels, values, color="#c44e52")
    ax.set_xlabel("feature importance" if hasattr(clf, "feature_importances_") else "|coefficient|")
    ax.set_title(f"Top {top} features ({type(clf).__name__})")
    return fig


def plot_keras_history(history) -> plt.Figure:
    """`history` may be a keras History object or a plain dict (from disk)."""
    h = history.history if hasattr(history, "history") else history
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(h["loss"], label="train")
    axes[0].plot(h["val_loss"], label="validation")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("binary cross-entropy")
    axes[0].set_title("MLP training loss"); axes[0].legend()
    axes[1].plot(h["auc"], label="train")
    axes[1].plot(h["val_auc"], label="validation")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("AUC")
    axes[1].set_title("MLP validation AUC (early stopping)"); axes[1].legend()
    fig.tight_layout()
    return fig


def metrics_table(metrics: dict) -> str:
    """Markdown comparison table from the metrics.json structure."""
    lines = [
        "| Model | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 | Fit time |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, m in metrics["models"].items():
        label = MODEL_LABELS.get(name, name)
        if name == metrics.get("best_model"):
            label = f"**{label}**"
        lines.append(
            f"| {label} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | {m['accuracy']:.4f} "
            f"| {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} "
            f"| {metrics['fit_seconds'][name]:.0f}s |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def generate_all_from_disk() -> None:
    """CLI path: rebuild every figure from models/ artifacts."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    reports_dir = REPO_ROOT / "reports"

    with open(MODELS_DIR / "metrics.json") as fh:
        metrics = json.load(fh)
    preds = np.load(MODELS_DIR / "preds_test.npz")
    y_test = preds["y_test"]
    probas = {k: preds[k] for k in preds.files if k != "y_test"}

    plot_roc_curves(y_test, probas).savefig(FIG_DIR / "08_roc_curves.png"); plt.close("all")
    plot_pr_curves(y_test, probas).savefig(FIG_DIR / "09_pr_curves.png"); plt.close("all")
    plot_confusion_matrices(y_test, probas).savefig(FIG_DIR / "10_confusion_matrices.png"); plt.close("all")

    xgb = joblib.load(MODELS_DIR / "xgboost.joblib")
    plot_feature_importance(xgb).savefig(FIG_DIR / "11_feature_importance.png"); plt.close("all")

    with open(MODELS_DIR / "keras_history.json") as fh:
        history = json.load(fh)
    plot_keras_history(history).savefig(FIG_DIR / "12_keras_training_history.png"); plt.close("all")

    table = metrics_table(metrics)
    (reports_dir / "model_comparison.md").write_text(
        "# Model comparison — test set (20% hold-out)\n\n" + table + "\n")
    print("Evaluation figures + model_comparison.md written to reports/")
    print(table)


if __name__ == "__main__":
    generate_all_from_disk()
