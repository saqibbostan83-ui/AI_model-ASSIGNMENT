"""
Preprocessing / feature engineering for the Diabetes 130-US Hospitals dataset.

Pipeline stages
---------------
1. load_raw()          - read CSV, replace '?' with proper NaN
2. clean_basic()       - readable, decoded dataframe for EDA:
                         * IDs_mapping decode of admission/discharge/source IDs
                         * ICD-9 diagnosis codes -> 18 disease groups
                         * age brackets -> numeric midpoints
3. build_features()    - model-ready dataframe:
                         * ordinal-encode medications / lab results
                         * drop identifier & useless columns
                         * binary target: readmitted within 30 days
4. get_dataset()       - one-call convenience: cleaned data + stratified split
5. build_preprocessor()- sklearn ColumnTransformer (OHE categoricals,
                         scale numerics) used inside every model pipeline

Design note: ALL 101,766 encounters are kept (including a few thousand
discharges to hospice/expired), so the dataset stays above the assignment's
100,000-row requirement. The model can learn that such patients are never
re-admitted.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = REPO_ROOT / "data" / "raw" / "diabetic_data.csv"
IDS_CSV = REPO_ROOT / "data" / "raw" / "IDs_mapping.csv"

TARGET = "readmitted"          # original 3-level column
TARGET_BIN = "readmitted_30d"  # engineered binary target

RANDOM_STATE = 42

# --------------------------------------------------------------------------- #
# 1. Raw loading
# --------------------------------------------------------------------------- #
def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    """Read the raw CSV and replace the dataset's '?' missing-code with NaN."""
    df = pd.read_csv(path)
    df = df.replace("?", np.nan)
    return df


# --------------------------------------------------------------------------- #
# 2. IDs_mapping decoding
# --------------------------------------------------------------------------- #
def _parse_ids_mapping(path: Path = IDS_CSV) -> Dict[str, Dict[int, str]]:
    """IDs_mapping.csv is really 3 small tables stacked with blank lines."""
    sections: Dict[str, Dict[int, str]] = {}
    current_name, current = None, {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            first, _, second = line.partition(",")
            if second in ("description", ""):           # header row
                if current_name is not None:
                    sections[current_name] = current
                current_name, current = first, {}
            elif first.replace("-", "").isdigit():
                current[int(first)] = second
            else:                                       # blank description
                current[int(first)] = "Unknown"
    if current_name is not None:
        sections[current_name] = current
    return sections


_IDS = _parse_ids_mapping()

_ADM_TYPE_MAP = _IDS.get("admission_type_id", {})
_DISCHARGE_MAP = _IDS.get("discharge_disposition_id", {})
_ADM_SOURCE_MAP = _IDS.get("admission_source_id", {})

# Collapse 'Not Available' / 'NULL' / 'Not Mapped' style labels
_UNK_LABELS = {"Not Available", "Unknown", "NULL", "Not Mapped", ""}


def _map_or_unknown(mapping: Dict[int, str], key) -> str:
    label = mapping.get(int(key), "Unknown") if str(key).replace("-", "").isdigit() else "Unknown"
    return "Unknown" if label in _UNK_LABELS else label


def _group_discharge(desc: str) -> str:
    """Group the ~29 discharge dispositions into 6 clinically meaningful buckets."""
    d = desc.lower()
    if "expired" in d or "hospice" in d:
        return "Expired/Hospice"
    if d.startswith("discharged to home"):
        return "Home"
    if "transfer" in d:
        return "Transferred"
    if "another" in d or "short-term" in d or "snf" in d or "icf" in d \
            or "facility" in d or "hospital" in d or "care" in d:
        return "Another facility"
    if "left ama" in d or "still patient" in d or "outpatient" in d:
        return "Left AMA/Other"
    return "Other/Unknown"


def _group_source(desc: str) -> str:
    """Group the ~26 admission sources into 6 buckets."""
    d = desc.lower()
    if "emergency room" in d:
        return "Emergency Room"
    if "referral" in d or "clinic" in d or "physician" in d or "hmo" in d:
        return "Referral/Clinic"
    if "transfer" in d or "snf" in d or "hospital" in d or "icf" in d or "facility" in d:
        return "Transfer"
    if d in ("court", "law enforcement"):
        return "Court/Law"
    if "born" in d:
        return "Born in hospital"
    return "Other/Unknown"


# --------------------------------------------------------------------------- #
# 3. ICD-9 grouping
# --------------------------------------------------------------------------- #
def map_icd9(code) -> str:
    """Map one ICD-9 diagnosis code to a disease group (Strack et al. 2014)."""
    if code is None or (isinstance(code, float) and np.isnan(code)) or str(code).strip() in ("", "nan", "None"):
        return "Unknown"
    code = str(code).strip()
    if code.startswith(("E", "V")):
        return "Other"
    try:
        c = float(code.split(".")[0])
    except ValueError:
        return "Other"
    if 250 <= c < 251:
        return "Diabetes"
    if 390 <= c <= 459 or c == 785:
        return "Circulatory"
    if 460 <= c <= 519 or c == 786:
        return "Respiratory"
    if 520 <= c <= 579 or c == 787:
        return "Digestive"
    if 800 <= c <= 999:
        return "Injury"
    if 710 <= c <= 739:
        return "Musculoskeletal"
    if 580 <= c <= 629 or c == 788:
        return "Genitourinary"
    if 140 <= c <= 239:
        return "Neoplasms"
    if 290 <= c <= 319:
        return "Mental"
    if 280 <= c <= 289:
        return "Blood"
    if 320 <= c <= 359:
        return "Nervous"
    if 630 <= c <= 679:
        return "Pregnancy"
    if 680 <= c <= 709 or c == 783:
        return "Skin"
    if 360 <= c <= 389:
        return "Sense organs"
    if 1 <= c <= 139:
        return "Infectious"
    return "Other"


# --------------------------------------------------------------------------- #
# 4. Column groups
# --------------------------------------------------------------------------- #
ID_COLS = ["encounter_id", "patient_nbr"]
DROP_COLS = ["weight", "payer_code", "citoglipton", "examide"]
# weight: ~97% missing | payer_code: ~40% missing, weak signal
# citoglipton & examide: constant 'No' for every row

MEDICATION_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]
# examide/citoglipton excluded (constant) -> dropped outright

MED_ORDINAL = {"No": 0, "Down": 1, "Steady": 2, "Up": 3}
GLU_ORDINAL = {"None": 0, "Norm": 1, ">200": 2, ">300": 3}
A1C_ORDINAL = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}
AGE_MIDPOINT = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45,
    "[50-60)": 55, "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95,
}

NUMERIC_FEATURES = [
    "age", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses", "service_utilization",
    "n_active_medications", "lab_monitoring_score",
]
CATEGORICAL_FEATURES = [
    "race", "gender", "medical_specialty", "admission_type",
    "discharge_disposition", "admission_source", "primary_diag",
    "secondary_diag", "additional_diag",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# --------------------------------------------------------------------------- #
# 5. Cleaning (readable dataframe used for EDA)
# --------------------------------------------------------------------------- #
def clean_basic(df: pd.DataFrame, top_specialties: int = 20) -> pd.DataFrame:
    """Decode IDs, group diagnoses, midpoint ages. Returns a readable frame."""
    out = df.copy()

    # --- demographics ---
    out["race"] = out["race"].fillna("Unknown")
    out["gender"] = out["gender"].fillna("Unknown")
    out["age"] = out["age"].map(AGE_MIDPOINT).astype(float)

    # --- decoded admission info ---
    out["admission_type"] = out["admission_type_id"].map(lambda k: _map_or_unknown(_ADM_TYPE_MAP, k))
    out["discharge_disposition"] = out["discharge_disposition_id"].map(lambda k: _map_or_unknown(_DISCHARGE_MAP, k)).map(_group_discharge)
    out["admission_source"] = out["admission_source_id"].map(lambda k: _map_or_unknown(_ADM_SOURCE_MAP, k)).map(_group_source)

    # --- medical specialty: keep top-N, rest 'Other' ---
    out["medical_specialty"] = out["medical_specialty"].fillna("Missing")
    top = out["medical_specialty"].value_counts().head(top_specialties).index
    out["medical_specialty"] = out["medical_specialty"].where(out["medical_specialty"].isin(top), "Other")

    # --- ICD-9 -> disease groups ---
    for col, new in [("diag_1", "primary_diag"), ("diag_2", "secondary_diag"), ("diag_3", "additional_diag")]:
        out[new] = out[col].map(map_icd9)

    # --- utilisation composite (standard in readmission literature) ---
    out["service_utilization"] = (out["number_outpatient"] + out["number_emergency"] + out["number_inpatient"])

    return out


# --------------------------------------------------------------------------- #
# 6. Model-ready features
# --------------------------------------------------------------------------- #
def build_features(df_clean: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) ready for the sklearn ColumnTransformer."""
    out = df_clean.copy()

    # ordinal encodings
    for col in MEDICATION_COLS:
        out[col] = out[col].map(MED_ORDINAL).fillna(0).astype(int)
    out["max_glu_serum"] = out["max_glu_serum"].fillna("None").map(GLU_ORDINAL).fillna(0).astype(int)
    out["A1Cresult"] = out["A1Cresult"].fillna("None").map(A1C_ORDINAL).fillna(0).astype(int)
    out["change"] = (out["change"] == "Ch").astype(int)
    out["diabetesMed"] = (out["diabetesMed"] == "Yes").astype(int)

    # compact medication summaries (feature engineering)
    med_matrix = out[MEDICATION_COLS]
    out["n_active_medications"] = (med_matrix > 0).sum(axis=1)            # count of meds prescribed
    out["lab_monitoring_score"] = out["max_glu_serum"] + out["A1Cresult"]  # 0 = glucose & A1C never measured

    # binary target: readmitted within 30 days
    y = (out[TARGET] == "<30").astype(int).rename(TARGET_BIN)

    X = out[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].fillna("Unknown").astype(str)
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """One-hot-encode categoricals, standard-scale numerics."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_dataset(test_size: float = 0.2) -> Dict:
    """One-call convenience: raw, cleaned, features and a stratified split."""
    raw = load_raw()
    clean = clean_basic(raw)
    X, y = build_features(clean)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    return {
        "raw": raw,
        "clean": clean,
        "X": X,
        "y": y,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
    }


if __name__ == "__main__":
    data = get_dataset()
    print("raw shape   :", data["raw"].shape)
    print("clean shape :", data["clean"].shape)
    print("X shape     :", data["X"].shape)
    print("target rate : {:.2%}".format(data["y"].mean()))
    print("train/test  : {} / {}".format(data["X_train"].shape[0], data["X_test"].shape[0]))
