"""
Builds notebooks/readmission_modeling.ipynb programmatically (nbformat),
then it can be executed with:

    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=1800 notebooks/readmission_modeling.ipynb

The notebook reuses the same src/ modules as the CLI scripts, so results are
reproducible across the whole repo.
"""
from pathlib import Path

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parents[1]
NB_PATH = REPO_ROOT / "notebooks" / "readmission_modeling.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.11"}

MD = nbf.v4.new_markdown_cell
CODE = nbf.v4.new_code_cell

cells = []

# ------------------------------------------------------------------ #
cells.append(MD(
    "# Predicting 30-Day Hospital Readmission for Diabetes Patients\n"
    "\n"
    "**Data Science assignment — end-to-end machine learning project**\n"
    "\n"
    "| | |\n"
    "|---|---|\n"
    "| **Dataset** | Diabetes 130-US Hospitals for years 1999–2008 (UCI ML Repository, ID 296) |\n"
    "| **Size** | 101,766 encounters × 50 raw columns ✔ (requirement: >100,000 rows, ≥35 columns) |\n"
    "| **Task** | Binary classification — will the patient be re-admitted within 30 days? |\n"
    "| **Models** | Logistic Regression · Random Forest · XGBoost · Keras neural network (+ SMOTE comparison) |\n"
    "\n"
    "Hospital readmission shortly after discharge is expensive and often preventable. "
    "US hospitals are financially penalised for excess 30-day readmissions, so flagging "
    "high-risk diabetes patients at discharge time lets care teams target follow-up visits. "
    "\n"
    "The dataset covers 10 years of clinical encounters at 130 US hospitals and was "
    "curated by Strack et al. (2014). It satisfies the assignment requirements with "
    "**101,766 rows and 50 columns**."
))

# ------------------------------------------------------------------ #
cells.append(MD("## 1 · Setup"))
cells.append(CODE(
    "import sys, os\n"
    "from pathlib import Path\n"
    "\n"
    "# The notebook lives in notebooks/ — switch to the repo root and import the\n"
    "# same src/ modules the CLI scripts use (single source of truth).\n"
    "if not Path('src').exists():\n"
    "    os.chdir('..')\n"
    "sys.path.insert(0, os.getcwd())\n"
    "\n"
    "import json\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "import sklearn, xgboost, tensorflow as tf\n"
    "\n"
    "from src.preprocess import get_dataset, load_raw, clean_basic, build_features\n"
    "import src.eda as eda\n"
    "import src.evaluate as ev\n"
    "from src.train import train_all_models\n"
    "\n"
    "sns.set_theme(style='whitegrid', palette='deep')\n"
    "plt.rcParams.update({'figure.dpi': 100, 'savefig.bbox': 'tight'})\n"
    "pd.set_option('display.max_columns', 60)\n"
    "\n"
    "print('pandas ', pd.__version__)\n"
    "print('sklearn', sklearn.__version__)\n"
    "print('xgboost', xgboost.__version__)\n"
    "print('tensorflow', tf.__version__)"
))

# ------------------------------------------------------------------ #
cells.append(MD("## 2 · Load the data"))
cells.append(CODE(
    "raw = load_raw()\n"
    "print(f'{raw.shape[0]:,} encounters x {raw.shape[1]} columns')\n"
    "raw.head()"
))
cells.append(CODE(
    "# The dataset uses '?' as its missing-value code — replaced with NaN on load.\n"
    "raw.isna().sum().sort_values(ascending=False).head(10)"
))

# ------------------------------------------------------------------ #
cells.append(MD(
    "## 3 · Exploratory Data Analysis\n"
    "\n"
    "### 3.1 The target is heavily imbalanced"))
cells.append(CODE(
    "fig = eda.plot_target_distribution(raw)\n"
    "plt.show()"
))
cells.append(CODE(
    "fig = eda.plot_missing_values(raw)\n"
    "plt.show()"
))
cells.append(MD(
    "**Observations**\n"
    "- Only **11.2%** of encounters end in a readmission within 30 days → accuracy is a "
    "misleading metric; we will optimise/report **ROC-AUC and PR-AUC**.\n"
    "- `weight` is missing for 97% of encounters → dropped. `payer_code` (40%) and "
    "`medical_specialty` (49%) are kept/patched as an explicit `Missing`/`Other` category."
))
cells.append(MD("### 3.2 Numeric feature distributions"))
cells.append(CODE(
    "data = get_dataset()          # cleaned frame + stratified 80/20 split\n"
    "df = data['clean']            # readable version for EDA\n"
    "fig = eda.plot_numeric_distributions(df)\n"
    "plt.show()"
))
cells.append(MD("### 3.3 What actually drives readmission"))
cells.append(CODE(
    "fig = eda.plot_readmission_by_key_features(df)\n"
    "plt.show()"
))
cells.append(MD(
    "**Observations** — the early-readmission rate\n"
    "- rises with **age**;\n"
    "- is *highest for the shortest stays* (1–3 days — patients discharged perhaps too early);\n"
    "- explodes with **prior-year utilisation** (outpatient + emergency + inpatient visits), "
    "confirming that history is the strongest signal."
))
cells.append(CODE(
    "fig = eda.plot_correlation_heatmap(df)\n"
    "plt.show()"
))
cells.append(CODE(
    "fig = eda.plot_categorical_target(df)\n"
    "plt.tight_layout(); plt.show()"
))
cells.append(CODE(
    "fig = eda.plot_medication_usage(df)\n"
    "plt.show()"
))

# ------------------------------------------------------------------ #
cells.append(MD(
    "## 4 · Preprocessing & feature engineering\n"
    "\n"
    "Applied by `src/preprocess.py`:\n"
    "\n"
    "| Step | Detail |\n"
    "|---|---|\n"
    "| Drop | `encounter_id`, `patient_nbr` (identifiers), `weight` (97% missing), `payer_code` (40% missing, weak), `citoglipton`/`examide` (constant) |\n"
    "| Decode | `admission/discharge/source IDs` → clinical categories via `IDs_mapping.csv`; ~700 ICD-9 codes → 17 disease groups |\n"
    "| Encode | age brackets → midpoints; 21 medication columns → ordinal {No, Down, Steady, Up}; HbA1c/glucose → ordinal severity |\n"
    "| Engineer | `service_utilization` (prior visits), `n_active_medications`, `lab_monitoring_score` |\n"
    "| Target | `readmitted` → binary: `<30` = 1, else 0 |\n"
    "\n"
    "All **101,766 encounters are kept** (the assignment requires >100,000 rows), "
    "including hospice/expired discharges — the model can learn these are never readmitted."
))
cells.append(CODE(
    "X, y = build_features(data['clean'])\n"
    "print(f'features: {X.shape[1]}  |  rows: {X.shape[0]:,}  |  positive rate: {y.mean():.2%}')\n"
    "X.head()"
))
cells.append(CODE(
    "print('train:', data['X_train'].shape, ' test:', data['X_test'].shape)\n"
    "data['y_train'].value_counts(normalize=True).rename('train').to_frame().join("
    "data['y_test'].value_counts(normalize=True).rename('test'))"
))

# ------------------------------------------------------------------ #
cells.append(MD(
    "## 5 · Modelling\n"
    "\n"
    "Four model families are compared on the same stratified 80/20 split "
    "(`random_state=42`):\n"
    "\n"
    "1. **Logistic Regression** — interpretable linear baseline (class-weighted)\n"
    "2. **Random Forest** — bagged trees (class-weighted)\n"
    "3. **XGBoost** — gradient boosting (`scale_pos_weight` ≈ 8 for imbalance)\n"
    "4. **Keras MLP** — 256→128→64 neural network with dropout + early stopping\n"
    "5. **SMOTE + LogReg** — oversampling alternative to class weights\n"
    "\n"
    "Hyperparameters for the Random Forest and XGBoost were selected with a small "
    "`RandomizedSearchCV` (3-fold, ROC-AUC) in `src/train.py`; this notebook uses the "
    "winning settings so it runs in minutes."
))
cells.append(CODE(
    "results = train_all_models(data, tune=False)"
))

# ------------------------------------------------------------------ #
cells.append(MD("## 6 · Evaluation"))
cells.append(CODE(
    "rows = {n: results[n]['metrics'] for n in results if not n.startswith('_')}\n"
    "summary = pd.DataFrame(rows).T[['roc_auc', 'pr_auc', 'accuracy', 'precision', 'recall', 'f1', 'f1_best_threshold']]\n"
    "summary.sort_values('roc_auc', ascending=False).style.format('{:.4f}').background_gradient(subset=['roc_auc', 'pr_auc'], cmap='Greens')"
))
cells.append(MD("### 6.1 ROC and Precision-Recall curves"))
cells.append(CODE(
    "y_test = data['y_test'].values\n"
    "probas = {n: results[n]['proba_test'] for n in results if not n.startswith('_')}\n"
    "fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))\n"
    "plt.sca(ax[0]); ev.plot_roc_curves(y_test, probas)\n"
    "plt.sca(ax[1]); ev.plot_pr_curves(y_test, probas)\n"
    "plt.tight_layout(); plt.show()"
))
cells.append(MD(
    "The **PR curve** is the honest view under an 11% positive rate: precision collapses "
    "as we push for high recall, so deployment threshold must be chosen per hospital "
    "priorities (missed patients vs. intervention capacity)."
))
cells.append(MD("### 6.2 Confusion matrices (0.5 threshold)"))
cells.append(CODE(
    "fig = ev.plot_confusion_matrices(y_test, probas)\n"
    "plt.show()"
))
cells.append(MD("### 6.3 What the model looks at — XGBoost feature importance"))
cells.append(CODE(
    "fig = ev.plot_feature_importance(results['xgboost']['model'])\n"
    "plt.show()"
))
cells.append(MD(
    "**Prior utilisation** (`number_inpatient`, `service_utilization`), **discharge "
    "disposition** and **diagnosis mix** dominate — exactly what the clinical literature "
    "reports for diabetic readmission."
))
cells.append(MD("### 6.4 Neural network training dynamics"))
cells.append(CODE(
    "fig = ev.plot_keras_history(results['keras_mlp']['history'])\n"
    "plt.show()"
))

# ------------------------------------------------------------------ #
cells.append(MD(
    "## 7 · Conclusions\n"
    "\n"
    "- Tree ensembles (XGBoost / Random Forest) beat the linear baseline and the neural "
    "network on this tabular, wide-but-shallow dataset — expected: ~100k rows of "
    "mostly categorical clinical features.\n"
    "- Class weighting handles the 11% positive rate at least as well as SMOTE, with "
    "far less compute.\n"
    "- With ROC-AUC around **0.69–0.70** the model is far better than chance (0.50) and "
    "in line with published results on this dataset; readmission is genuinely hard to "
    "predict from administrative data alone.\n"
    "- Next steps: richer feature engineering from medication changes, calibration "
    "analysis, cost-sensitive thresholds, and fairness checks across race/age groups.\n"
    "\n"
    "---\n"
    "*Reproduce: `python src/download_data.py && python src/train.py` — a live demo is "
    "available via `streamlit run app/app.py`.*"
))

nb.cells = cells

if __name__ == "__main__":
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NB_PATH)
    print(f"wrote {NB_PATH} with {len(cells)} cells")
