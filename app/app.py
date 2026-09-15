"""
Streamlit demo: 30-day readmission risk for diabetes patients.

Run with:
    streamlit run app/app.py

Uses the best model trained by src/train.py (models/best_model.joblib —
a full sklearn pipeline that accepts the cleaned feature dataframe).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

import joblib
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"

st.set_page_config(page_title="Readmission Risk", page_icon="🏥", layout="wide")


# ------------------------------------------------------------------ #
# Cached resources
# ------------------------------------------------------------------ #
@st.cache_resource
def load_artifacts():
    pipeline = joblib.load(MODELS_DIR / "best_model.joblib")
    with open(MODELS_DIR / "default_patient.json") as fh:
        defaults = json.load(fh)
    with open(MODELS_DIR / "metrics.json") as fh:
        metrics = json.load(fh)
    return pipeline, defaults, metrics


# ------------------------------------------------------------------ #
# Page
# ------------------------------------------------------------------ #
st.title("🏥 Diabetes Patient Readmission Risk")
st.caption(
    "Predicts the probability that a diabetes patient is **re-admitted to hospital "
    "within 30 days** of discharge, using a model trained on 101,766 US hospital "
    "encounters (1999–2008)."
)

try:
    pipeline, defaults, metrics = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts not found in `models/`. Run `python src/train.py` first.")
    st.stop()

base_rate = metrics["dataset"]["positive_rate"]
best_name = metrics["best_model"]
best_metrics = metrics["models"][best_name]

with st.sidebar:
    st.header("Model card")
    st.markdown(f"""
**Best model:** `{best_name}`
**Test ROC-AUC:** {best_metrics['roc_auc']:.3f}
**Test PR-AUC:** {best_metrics['pr_auc']:.3f}
**Recall (30d):** {best_metrics['recall']:.1%}

Training data: {metrics['dataset']['rows']:,} encounters
{metrics['dataset']['columns_raw']} raw columns
Positive class rate: {base_rate:.1%}

Models compared: {len(metrics['models'])}
(""")
    st.divider()
    st.markdown(
        "*This demo is for educational purposes only and is **not** a medical "
        "device.* Do not use it for real clinical decisions."
    )

st.markdown(
    "Edit the patient profile below and press **Predict**. Fields you leave "
    "untouched keep the training-set average value."
)

# ------------------------------------------------------------------ #
# Input form
# ------------------------------------------------------------------ #
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Patient")
    age = st.slider("Age (years)", 5, 95, int(defaults.get("age", 65)), 5)
    gender = st.selectbox("Gender", ["Female", "Male", "Unknown/Invalid"],
                          index=["Female", "Male", "Unknown/Invalid"].index(defaults.get("gender", "Female")))
    race = st.selectbox("Race",
                        ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Unknown"],
                        index=["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Unknown"].index(defaults.get("race", "Caucasian")))
    a1c = st.selectbox("HbA1c result (last test)", ["None", "Norm", ">7", ">8"],
                       index=["None", "Norm", ">7", ">8"].index(defaults.get("A1Clabel", "None")))
    glu = st.selectbox("Glucose serum result", ["None", "Norm", ">200", ">300"], index=0)
    on_diabetes_meds = st.radio("Prescribed diabetes medication?", ["Yes", "No"], index=0)

with col2:
    st.subheader("This admission")
    adm_type = st.selectbox("Admission type",
                            ["Emergency", "Urgent", "Elective", "Newborn", "Trauma Center", "Unknown"])
    adm_source = st.selectbox("Admission source",
                              ["Emergency Room", "Referral/Clinic", "Transfer", "Other/Unknown"])
    time_in_hospital = st.slider("Length of stay (days)", 1, 14, int(defaults.get("time_in_hospital", 4)))
    n_diagnoses = st.slider("Number of diagnoses", 1, 16, int(defaults.get("number_diagnoses", 7)))
    n_lab_proc = st.slider("Lab procedures performed", 0, 120, int(defaults.get("num_lab_procedures", 44)))
    n_procedures = st.slider("Other procedures", 0, 6, int(defaults.get("num_procedures", 1)))

with col3:
    st.subheader("History & discharge")
    discharge = st.selectbox("Discharged to",
                             ["Home", "Another facility", "Transferred", "Left AMA/Other", "Other/Unknown", "Expired/Hospice"])
    n_outpatient = st.slider("Outpatient visits (prior year)", 0, 30, int(defaults.get("number_outpatient", 0)))
    n_emergency = st.slider("Emergency visits (prior year)", 0, 30, int(defaults.get("number_emergency", 0)))
    n_inpatient = st.slider("Inpatient visits (prior year)", 0, 20, int(defaults.get("number_inpatient", 0)))
    n_medications = st.slider("Medications during stay", 0, 80, int(defaults.get("num_medications", 15)))
    primary_diag = st.selectbox("Primary diagnosis group",
                                ["Diabetes", "Circulatory", "Respiratory", "Digestive", "Genitourinary",
                                 "Injury", "Musculoskeletal", "Neoplasms", "Other", "Unknown"])

# ------------------------------------------------------------------ #
# Assemble the feature vector (cleaned-feature schema expected by the pipeline)
# ------------------------------------------------------------------ #
A1C_MAP = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}
GLU_MAP = {"None": 0, "Norm": 1, ">200": 2, ">300": 3}

patient = dict(defaults)  # start from training-set averages
patient.update({
    "age": float(age),
    "gender": gender,
    "race": race,
    "admission_type": adm_type,
    "admission_source": adm_source,
    "discharge_disposition": discharge,
    "time_in_hospital": time_in_hospital,
    "number_diagnoses": n_diagnoses,
    "num_lab_procedures": n_lab_proc,
    "num_procedures": n_procedures,
    "number_outpatient": n_outpatient,
    "number_emergency": n_emergency,
    "number_inpatient": n_inpatient,
    "num_medications": n_medications,
    "primary_diag": primary_diag,
    "service_utilization": n_outpatient + n_emergency + n_inpatient,
    "lab_monitoring_score": A1C_MAP[a1c] + GLU_MAP[glu],
    "diabetesMed": 1 if on_diabetes_meds == "Yes" else 0,
})

st.divider()

if st.button("🔮 Predict 30-day readmission risk", type="primary", use_container_width=True):
    X_input = pd.DataFrame([patient])[list(defaults.keys())]
    proba = float(pipeline.predict_proba(X_input)[0, 1])

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.metric("Predicted readmission risk", f"{proba:.1%}")
        st.progress(min(proba, 1.0))
        rel = proba / base_rate
        if rel >= 2:
            verdict, emoji = "well above average risk", "🔴"
        elif rel >= 1.25:
            verdict, emoji = "above average risk", "🟠"
        elif rel >= 0.75:
            verdict, emoji = "around average risk", "🟡"
        else:
            verdict, emoji = "below average risk", "🟢"
        st.markdown(
            f"{emoji} This profile is **{verdict}** — the model predicts "
            f"**{proba:.1%}** vs. a population base rate of **{base_rate:.1%}** "
            f"({rel:.1f}× the average patient)."
        )
    with c2:
        st.metric("Population base rate", f"{base_rate:.1%}")
    with c3:
        st.metric("Model ROC-AUC", f"{best_metrics['roc_auc']:.3f}")

    with st.expander("See the exact feature values used"):
        st.dataframe(X_input.T.rename(columns={0: "value"}), use_container_width=True)

    st.info(
        "Prediction computed by the best model from `src/train.py`, served as "
        "`models/best_model.joblib`. See the README for model details and caveats."
    )
else:
    st.markdown("⬆️ Set the patient profile, then press **Predict**.")
