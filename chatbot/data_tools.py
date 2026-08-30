import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# CACHED DATA
# ============================================================

_master = None
_activity = None
_lsr = None
_precursor = None
_barrier = None


# ============================================================
# LOAD DATA
# ============================================================

def load_master_data():
    global _master

    if _master is None:
        _master = pd.read_csv(
            ROOT / "SIF_Dashboard_Master_v4.csv",
            low_memory=False
        )

    return _master


def load_data():
    global _activity, _lsr, _precursor, _barrier

    if _activity is None:
        _activity = pd.read_csv(
            ROOT / "SIF_Activity_Ranking_v4.csv"
        )

    if _lsr is None:
        _lsr = pd.read_csv(
            ROOT / "SIF_LSR_Ranking_v4.csv"
        )

    if _precursor is None:
        _precursor = pd.read_csv(
            ROOT / "SIF_Precursor_Ranking_v4.csv"
        )

    if _barrier is None:
        _barrier = pd.read_csv(
            ROOT / "SIF_Barrier_Ranking_v4.csv"
        )

    return _activity, _lsr, _precursor, _barrier


# ============================================================
# JSON-SAFE CLEANING
# ============================================================

def clean_records(records):
    cleaned = []

    for record in records:
        item = {}

        for key, value in record.items():

            # Handle pandas missing values
            if pd.isna(value):
                item[key] = None

            # Handle infinity / -infinity
            elif isinstance(value, float) and not math.isfinite(value):
                item[key] = None

            # Convert numpy scalar values to normal Python values
            elif hasattr(value, "item"):
                try:
                    item[key] = value.item()
                except Exception:
                    item[key] = value

            else:
                item[key] = value

        cleaned.append(item)

    return cleaned


# ============================================================
# 1. DATASET STATISTICS
# ============================================================

def dataset_statistics():
    """
    Return overall statistics for the OIL SIF dataset.

    Includes:
    - total reports
    - SIF Potential reports
    - Non-SIF Potential reports
    - SIF percentage
    - average SIF probability
    """

    df = load_master_data()

    total_reports = len(df)

    result = {
        "total_reports": int(total_reports)
    }

    if "sif_prediction" in df.columns:

        counts = (
            df["sif_prediction"]
            .fillna("")
            .astype(str)
            .value_counts()
        )

        sif_count = int(
            counts.get("SIF Potential", 0)
        )

        non_sif_count = int(
            counts.get("Non-SIF Potential", 0)
        )

        result["sif_potential"] = sif_count
        result["non_sif_potential"] = non_sif_count

        if total_reports > 0:
            result["sif_percentage"] = round(
                sif_count / total_reports * 100,
                2
            )
        else:
            result["sif_percentage"] = 0.0

    if "sif_probability" in df.columns:

        probabilities = pd.to_numeric(
            df["sif_probability"],
            errors="coerce"
        )

        mean_probability = probabilities.mean()

        if pd.isna(mean_probability):
            result["average_sif_probability"] = None
        else:
            result["average_sif_probability"] = round(
                float(mean_probability),
                4
            )

    return result


# ============================================================
# 2. FIND INCIDENTS
# ============================================================

def find_incidents(
    search_term: str,
    limit: int = 10
):
    """
    Find incidents related to a concept in the dataset.

    Gemini decides what concept to search for.
    This function only performs the dataset search.
    """

    if not search_term:
        return []

    df = load_master_data()

    search_term = str(search_term).strip().lower()

    if not search_term:
        return []

    # Prevent unreasonable tool requests
    limit = max(1, min(int(limit), 50))

    searchable_columns = [
        "report_text",
        "activity_group",
        "precursor",
        "life_saving_rule",
        "job",
        "outcome"
    ]

    searchable_columns = [
        column
        for column in searchable_columns
        if column in df.columns
    ]

    if not searchable_columns:
        return []

    mask = pd.Series(
        False,
        index=df.index
    )

    for column in searchable_columns:

        values = (
            df[column]
            .fillna("")
            .astype(str)
            .str.lower()
        )

        mask |= values.str.contains(
            search_term,
            regex=False
        )

    results = (
        df[mask]
        .head(limit)
    )

    return clean_records(
        results.to_dict(
            orient="records"
        )
    )


# ============================================================
# 3. GET INCIDENT BY REPORT ID
# ============================================================

def get_incident_by_id(
    report_id: str
):
    """
    Return a specific incident using its report ID.
    """

    if not report_id:
        return {
            "found": False,
            "report_id": None
        }

    df = load_master_data()

    report_id = (
        str(report_id)
        .strip()
        .upper()
    )

    if "report_id" not in df.columns:
        return {
            "found": False,
            "report_id": report_id,
            "error": "report_id column not found"
        }

    ids = (
        df["report_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    result = df[ids == report_id]

    if result.empty:
        return {
            "found": False,
            "report_id": report_id
        }

    return {
        "found": True,
        "report": clean_records(
            result.head(1).to_dict(
                orient="records"
            )
        )[0]
    }


# ============================================================
# GENERIC RANKING HELPER
# ============================================================

def _ranking(
    dataframe,
    n: int,
    metric: str
):
    allowed_metrics = {
        "sif_density_pct",
        "sif_reports",
        "total_reports"
    }

    if metric not in allowed_metrics:
        metric = "sif_density_pct"

    if metric not in dataframe.columns:
        return []

    n = max(1, min(int(n), 20))

    result = (
        dataframe
        .sort_values(
            metric,
            ascending=False
        )
        .head(n)
    )

    return clean_records(
        result.to_dict(
            orient="records"
        )
    )


# ============================================================
# 4. TOP ACTIVITIES
# ============================================================

def top_activities(
    n: int = 5,
    metric: str = "sif_density_pct"
):
    """
    Return the top activity groups.

    metric can be:
    - sif_density_pct
    - sif_reports
    - total_reports
    """

    activity, _, _, _ = load_data()

    return _ranking(
        activity,
        n,
        metric
    )


# ============================================================
# 5. TOP PRECURSORS
# ============================================================

def top_precursors(
    n: int = 5,
    metric: str = "sif_density_pct"
):
    """
    Return the top precursor categories.

    metric can be:
    - sif_density_pct
    - sif_reports
    - total_reports
    """

    _, _, precursor, _ = load_data()

    return _ranking(
        precursor,
        n,
        metric
    )


# ============================================================
# 6. TOP BARRIERS
# ============================================================

def top_barriers(
    n: int = 5,
    metric: str = "sif_density_pct"
):
    """
    Return the top barrier themes.

    metric can be:
    - sif_density_pct
    - sif_reports
    - total_reports
    """

    _, _, _, barrier = load_data()

    return _ranking(
        barrier,
        n,
        metric
    )


# ============================================================
# 7. TOP LIFE-SAVING RULES
# ============================================================

def top_life_saving_rules(
    n: int = 5,
    metric: str = "sif_density_pct"
):
    """
    Return the top Life-Saving Rules.

    metric can be:
    - sif_density_pct
    - sif_reports
    - total_reports
    """

    _, lsr, _, _ = load_data()

    return _ranking(
        lsr,
        n,
        metric
    )