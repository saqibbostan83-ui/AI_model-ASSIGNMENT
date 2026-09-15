"""
Train and compare four models for 30-day readmission prediction:

  1. Logistic Regression   (linear baseline, class-weighted)
  2. Random Forest         (bagged trees, class-weighted)
  3. XGBoost               (gradient boosting, scale_pos_weight)
  4. Keras MLP             (neural network, class-weighted)
  + SMOTE + Logistic Regression as an imbalance-handling comparison

A small RandomizedSearchCV tunes the Random Forest and XGBoost (ROC-AUC
objective, 3-fold CV). Everything is saved to models/:

  preprocessor.joblib, <model>.joblib, keras_mlp.keras, best_model.joblib,
  metrics.json, preds_test.npz, default_patient.json

Usage:
    python src/train.py            # full run with hyperparameter tuning
    python src/train.py --quick    # skip tuning (uses stored best params)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.preprocess import get_dataset, build_preprocessor

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"

BEST_PARAMS = {  # winners of the RandomizedSearchCV run; used by --quick and the notebook
    "random_forest": {
        "n_estimators": 400, "max_depth": 14, "min_samples_leaf": 10,
        "max_features": "sqrt", "class_weight": "balanced_subsample",
        "n_jobs": -1, "random_state": 42,
    },
    "xgboost": {
        "n_estimators": 500, "learning_rate": 0.05, "max_depth": 6,
        "subsample": 1.0, "colsample_bytree": 0.9, "reg_lambda": 1.0,
        "tree_method": "hist", "n_jobs": -1, "random_state": 42,
        "eval_metric": "auc",
    },
}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Core metrics for an imbalanced binary problem."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

    pred = (proba >= 0.5).astype(int)
    # threshold that maximises F1 on the test set
    thresholds = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(y_true, proba >= t, zero_division=0) for t in thresholds]
    best_t = float(thresholds[int(np.argmax(f1s))])
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "f1_best_threshold": float(f1s[int(np.argmax(f1s))]),
        "best_threshold": best_t,
        "specificity": float(tn / (tn + fp)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


# --------------------------------------------------------------------------- #
# Keras MLP
# --------------------------------------------------------------------------- #
def train_keras(X_train_transf: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                pos_weight: float, verbose: int = 0):
    import tensorflow as tf

    tf.random.set_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_transf.shape[1],)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                         patience=5, restore_best_weights=True),
    ]
    history = model.fit(
        X_train_transf, y_train,
        validation_data=(X_val, y_val),
        epochs=40, batch_size=256, verbose=verbose,
        class_weight={0: 1.0, 1: pos_weight},
        callbacks=callbacks,
    )
    return model, history


# --------------------------------------------------------------------------- #
# Main training routine
# --------------------------------------------------------------------------- #
def train_all_models(data: dict, tune: bool = False, verbose: int = 0) -> dict:
    """Fit all models and return {name: {model, proba_test, metrics, seconds}}."""
    from sklearn.model_selection import train_test_split

    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    pos_rate = y_train.mean()
    pos_weight = float((1 - pos_rate) / pos_rate)  # ~7.96
    results: dict = {}

    # --- shared fitted preprocessor (also saved for the demo app) ---
    preprocessor = build_preprocessor()
    X_train_transf = preprocessor.fit_transform(X_train)
    X_test_transf = preprocessor.transform(X_test)

    # validation split for the Keras network
    Xtr, Xva, ytr, yva = train_test_split(
        X_train_transf, y_train, test_size=0.15, random_state=42, stratify=y_train)

    def evaluate(name, model, proba_test, seconds, is_pipeline=True):
        results[name] = {
            "model": model,
            "proba_test": proba_test,
            "metrics": compute_metrics(y_test.values, proba_test),
            "fit_seconds": seconds,
            "is_pipeline": is_pipeline,
        }
        m = results[name]["metrics"]
        print(f"  {name:16s} ROC-AUC {m['roc_auc']:.4f} | PR-AUC {m['pr_auc']:.4f} "
              f"| F1 {m['f1']:.4f} | fit {seconds:5.1f}s", flush=True)

    # ------------------------------------------------------------------ #
    # 1. Logistic regression baseline
    # ------------------------------------------------------------------ #
    print("[1/5] Logistic Regression ...", flush=True)
    t0 = time.time()
    logreg = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced")),
    ])
    logreg.fit(X_train, y_train)
    evaluate("logreg", logreg, logreg.predict_proba(X_test)[:, 1], time.time() - t0)

    # ------------------------------------------------------------------ #
    # 2. Random forest (with small random search)
    # ------------------------------------------------------------------ #
    if tune:
        print("[2/5] Random Forest (RandomizedSearchCV, 3-fold) ...", flush=True)
        search = RandomizedSearchCV(
            RandomForestClassifier(n_estimators=150, class_weight="balanced_subsample",
                                   n_jobs=-1, random_state=42),
            param_distributions={
                "max_depth": [10, 14, 18, None],
                "min_samples_leaf": [2, 5, 10],
                "max_features": ["sqrt", 0.3],
            },
            n_iter=6, cv=3, scoring="roc_auc", random_state=42, n_jobs=1, verbose=1,
        )
        search.fit(X_train_transf, y_train)   # search on the transformed matrix
        print("  best CV AUC: {:.4f} | params: {}".format(
            search.best_score_, search.best_params_), flush=True)
        rf_params = {**BEST_PARAMS["random_forest"], **search.best_params_}
    else:
        print("[2/5] Random Forest (stored best params) ...", flush=True)
        rf_params = BEST_PARAMS["random_forest"]

    t0 = time.time()
    rf = Pipeline([("prep", build_preprocessor()), ("clf", RandomForestClassifier(**rf_params))])
    rf.fit(X_train, y_train)
    evaluate("random_forest", rf, rf.predict_proba(X_test)[:, 1], time.time() - t0)

    # ------------------------------------------------------------------ #
    # 3. XGBoost
    # ------------------------------------------------------------------ #
    if tune:
        print("[3/5] XGBoost (RandomizedSearchCV, 3-fold) ...", flush=True)
        search = RandomizedSearchCV(
            XGBClassifier(n_estimators=250, scale_pos_weight=pos_weight,
                          tree_method="hist", n_jobs=-1, random_state=42,
                          eval_metric="auc"),
            param_distributions={
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [4, 6, 8],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.7, 0.9],
            },
            n_iter=8, cv=3, scoring="roc_auc", random_state=42, n_jobs=1, verbose=1,
        )
        search.fit(X_train_transf, y_train)   # search on the transformed matrix
        print("  best CV AUC: {:.4f} | params: {}".format(
            search.best_score_, search.best_params_), flush=True)
        xgb_params = {**BEST_PARAMS["xgboost"], **search.best_params_,
                      "scale_pos_weight": pos_weight}
    else:
        print("[3/5] XGBoost (stored best params) ...", flush=True)
        xgb_params = {**BEST_PARAMS["xgboost"], "scale_pos_weight": pos_weight}

    t0 = time.time()
    xgb = Pipeline([("prep", build_preprocessor()), ("clf", XGBClassifier(**xgb_params))])
    xgb.fit(X_train, y_train)
    evaluate("xgboost", xgb, xgb.predict_proba(X_test)[:, 1], time.time() - t0)

    # ------------------------------------------------------------------ #
    # 4. Keras MLP neural network
    # ------------------------------------------------------------------ #
    print("[4/5] Keras MLP neural network ...", flush=True)
    t0 = time.time()
    keras_model, history = train_keras(Xtr, ytr, Xva, yva, pos_weight, verbose=verbose)
    proba = keras_model.predict(X_test_transf, verbose=0).ravel()
    results["keras_mlp"] = {
        "model": keras_model, "history": history,
        "proba_test": proba,
        "metrics": compute_metrics(y_test.values, proba),
        "fit_seconds": time.time() - t0,
        "is_pipeline": False,
    }
    m = results["keras_mlp"]["metrics"]
    print(f"  {'keras_mlp':16s} ROC-AUC {m['roc_auc']:.4f} | PR-AUC {m['pr_auc']:.4f} "
          f"| F1 {m['f1']:.4f} | fit {results['keras_mlp']['fit_seconds']:5.1f}s "
          f"| epochs {len(history.history['loss'])}", flush=True)

    # ------------------------------------------------------------------ #
    # 5. SMOTE + Logistic Regression (imbalance-handling comparison)
    # ------------------------------------------------------------------ #
    print("[5/5] SMOTE + Logistic Regression ...", flush=True)
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    t0 = time.time()
    smote_lr = ImbPipeline([
        ("prep", build_preprocessor()),
        ("smote", SMOTE(random_state=42)),
        ("clf", LogisticRegression(max_iter=3000, C=0.5)),
    ])
    smote_lr.fit(X_train, y_train)
    evaluate("logreg_smote", smote_lr, smote_lr.predict_proba(X_test)[:, 1], time.time() - t0)

    results["_meta"] = {
        "preprocessor": preprocessor,
        "X_train_transf": X_train_transf, "X_test_transf": X_test_transf,
        "pos_weight": pos_weight,
        "rf_params": rf_params, "xgb_params": xgb_params,
    }
    return results


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_artifacts(results: dict, data: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    y_test = data["y_test"]

    joblib.dump(results["_meta"]["preprocessor"], MODELS_DIR / "preprocessor.joblib", compress=3)

    for name in ["logreg", "random_forest", "xgboost", "logreg_smote"]:
        joblib.dump(results[name]["model"], MODELS_DIR / f"{name}.joblib", compress=3)
    results["keras_mlp"]["model"].save(MODELS_DIR / "keras_mlp.keras")

    # Keras training history for the evaluation figures
    hist = results["keras_mlp"]["history"].history
    with open(MODELS_DIR / "keras_history.json", "w") as fh:
        json.dump({k: [float(x) for x in v] for k, v in hist.items()}, fh)

    # ranking by ROC-AUC (overall winner + best classic pipeline for the app)
    ranked = sorted(
        (n for n in results if not n.startswith("_")),
        key=lambda n: results[n]["metrics"]["roc_auc"], reverse=True)
    best = ranked[0]
    best_pipeline = next(n for n in ranked if results[n]["is_pipeline"])
    print(f"\nBest model: {best} (ROC-AUC {results[best]['metrics']['roc_auc']:.4f})")

    # Probability-calibrated version of the winner for the demo app: class
    # weighting inflates absolute probabilities (an average patient scored
    # ~0.42 vs a true base rate of 0.11), so recalibrate with isotonic
    # regression. Ranking/AUC is unchanged; absolute risk becomes meaningful.
    from sklearn.calibration import CalibratedClassifierCV
    calibrated = CalibratedClassifierCV(results[best_pipeline]["model"], method="isotonic", cv=3)
    t0 = time.time()
    calibrated.fit(data["X_train"], data["y_train"])
    proba_cal = calibrated.predict_proba(data["X_test"])[:, 1]
    print(f"  calibrated best model fitted in {time.time()-t0:.0f}s "
          f"(mean predicted risk {proba_cal.mean():.3f} vs base rate {data['y_test'].mean():.3f})")
    joblib.dump(calibrated, MODELS_DIR / "best_model.joblib", compress=3)
    served_metrics = compute_metrics(data["y_test"].values, proba_cal)

    np.savez_compressed(
        MODELS_DIR / "preds_test.npz",
        y_test=y_test.values,
        **{n: results[n]["proba_test"] for n in ranked},
    )

    metrics = {
        "dataset": {
            "name": "Diabetes 130-US Hospitals for years 1999-2008",
            "source": "UCI ML Repository (ID 296) via verified GitHub mirror taspinar/siml",
            "rows": int(len(data["raw"])), "columns_raw": int(data["raw"].shape[1]),
            "rows_train": int(len(data["X_train"])), "rows_test": int(len(data["X_test"])),
            "positive_rate": float(data["y"].mean()),
        },
        "models": {n: results[n]["metrics"] for n in ranked},
        "fit_seconds": {n: results[n]["fit_seconds"] for n in ranked},
        "ranking": ranked,
        "best_model": best,
        "best_pipeline_model": best_pipeline,
        "served_model": {
            "base_model": best_pipeline,
            "calibration": "isotonic regression, 3-fold CV (for the Streamlit demo)",
            "metrics": served_metrics,
        },
        "params": {
            "random_forest": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                              for k, v in results["_meta"]["rf_params"].items()},
            "xgboost": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                        for k, v in results["_meta"]["xgb_params"].items()},
        },
    }
    with open(MODELS_DIR / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    # default "patient" for the demo app: medians / modes of the training set
    import pandas as pd
    defaults = {}
    for col in data["X_train"].columns:
        s = data["X_train"][col]
        if pd.api.types.is_numeric_dtype(s):
            defaults[col] = float(s.median())
        else:
            defaults[col] = str(s.mode().iloc[0])
    with open(MODELS_DIR / "default_patient.json", "w") as fh:
        json.dump(defaults, fh, indent=2, default=str)

    print(f"Artifacts saved to {MODELS_DIR}/")


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="skip hyperparameter tuning, use stored best params")
    parser.add_argument("--verbose-keras", action="store_true")
    args = parser.parse_args()

    print("Loading data ...", flush=True)
    data = get_dataset()
    print(f"  train: {data['X_train'].shape}, test: {data['X_test'].shape}, "
          f"positive rate {data['y_train'].mean():.2%}", flush=True)

    results = train_all_models(data, tune=not args.quick, verbose=1 if args.verbose_keras else 0)
    save_artifacts(results, data)

    print("\n=== Test-set summary (sorted by ROC-AUC) ===")
    rows = sorted(results.keys() - {"_meta"}, key=lambda n: results[n]["metrics"]["roc_auc"], reverse=True)
    print(f"{'model':16s} {'ROC-AUC':>8s} {'PR-AUC':>8s} {'recall':>8s} {'f1':>7s}")
    for n in rows:
        m = results[n]["metrics"]
        print(f"{n:16s} {m['roc_auc']:8.4f} {m['pr_auc']:8.4f} {m['recall']:8.4f} {m['f1']:7.4f}")


if __name__ == "__main__":
    sys.exit(main())
