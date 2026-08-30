from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "January2015toNovember2025.csv"
INFERENCE_PATH = ROOT / "data" / "raw" / "SIF_Dashboard_Full_ITA_Inference_v2.csv"
MASTER_PATH = ROOT / "data" / "raw" / "SIF_Dashboard_Master_v4.csv"


def _read_csv(path: Path, usecols=None):
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    # The supplied files are readable as UTF-8. latin1 is a safe fallback for future files.
    try:
        return pd.read_csv(path, encoding="utf-8", usecols=usecols, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", usecols=usecols, low_memory=False)


RAW_COLS = [
    "ID", "EventDate", "Employer", "City", "State", "Zip", "Primary NAICS",
    "Hospitalized", "Amputation", "Loss of Eye", "Final Narrative", "Nature",
    "NatureTitle", "Part of Body Title", "Event", "EventTitle", "SourceTitle",
    "FederalState"
]

MASTER_COLS = [
    "report_id", "source", "report_text", "sif_probability", "sif_prediction",
    "activity_group", "precursor", "date_of_incident", "industry_description",
    "job_description", "incident_outcome", "type_of_incident", "nature_title_pred",
    "event_title_pred", "life_saving_rule", "confidence_band", "hse_review_status",
    "priority_score"
]


@st.cache_data(show_spinner="Loading OSHA incident data…")
def load_raw():
    df = _read_csv(RAW_PATH, RAW_COLS)
    df["State"] = df["State"].astype("string").str.strip().str.upper()
    df["EventDate"] = pd.to_datetime(df["EventDate"], errors="coerce")
    for c in ["Hospitalized", "Amputation", "Loss of Eye"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


@st.cache_data(show_spinner="Loading SIF inference data…")
def load_inference():
    df = _read_csv(INFERENCE_PATH, MASTER_COLS[:14])
    df = _clean_master(df)
    return df


@st.cache_data(show_spinner="Loading SIF master intelligence…")
def load_master():
    df = _read_csv(MASTER_PATH, MASTER_COLS)
    df = _clean_master(df)
    return df


def _clean_master(df):
    df = df.copy()
    # Excel/CSV BOM normalization.
    df.columns = [str(c).lstrip("\ufeffï»¿") for c in df.columns]
    for c in ["sif_probability", "priority_score"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    if "date_of_incident" in df.columns:
        df["date_of_incident"] = pd.to_datetime(df["date_of_incident"], errors="coerce")
    return df


def ensure_data():
    missing = [str(p) for p in [RAW_PATH, INFERENCE_PATH, MASTER_PATH] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required dataset(s):\n" + "\n".join(missing))
