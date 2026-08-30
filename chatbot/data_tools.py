import pandas as pd
from pathlib import Path
import math


ROOT = Path(__file__).resolve().parent.parent


# Load these once and keep them cached.
_activity = None
_lsr = None
_precursor = None
_barrier = None
_master = None


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


def get_dataset_summary():
    activity, lsr, precursor, barrier = load_data()

    return {
        "activities": activity.to_dict(orient="records"),
        "life_saving_rules": lsr.to_dict(orient="records"),
        "precursors": precursor.to_dict(orient="records"),
        "barriers": barrier.to_dict(orient="records"),
    }

def search_incidents(
    search_term: str,
    limit: int = 10
):

    df = load_master_data()

    search_term = search_term.lower().strip()

    search_columns = [
        "report_text",
        "activity_group",
        "precursor",
        "life_saving_rule",
        "job",
        "outcome"
    ]

    available_columns = [
        col for col in search_columns
        if col in df.columns
    ]

    if not available_columns:
        return []

    mask = pd.Series(False, index=df.index)

    for column in available_columns:
        mask |= (
            df[column]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.contains(
                search_term,
                regex=False
            )
        )

    results = df[mask].head(limit)

    # Convert NaN / NaT / infinity into JSON-safe values
    records = results.to_dict(orient="records")

    cleaned_records = []

    for record in records:

        cleaned_record = {}

        for key, value in record.items():

            if pd.isna(value):
                cleaned_record[key] = None

            elif isinstance(value, float) and not math.isfinite(value):
                cleaned_record[key] = None

            else:
                cleaned_record[key] = value

        cleaned_records.append(cleaned_record)

    return cleaned_records


def get_top_precursors(
    n: int = 5,
    metric: str = "sif_density_pct"
):
    _, _, precursor, _ = load_data()

    allowed_metrics = [
        "sif_density_pct",
        "sif_reports",
        "total_reports"
    ]

    if metric not in allowed_metrics:
        metric = "sif_density_pct"

    result = (
        precursor
        .sort_values(metric, ascending=False)
        .head(n)
    )

    return result.to_dict(orient="records")


def get_top_activities(
    n: int = 5,
    metric: str = "sif_density_pct"
):
    activity, _, _, _ = load_data()

    allowed_metrics = [
        "sif_density_pct",
        "sif_reports",
        "total_reports"
    ]

    if metric not in allowed_metrics:
        metric = "sif_density_pct"

    result = (
        activity
        .sort_values(metric, ascending=False)
        .head(n)
    )

    return result.to_dict(orient="records")


def get_top_barriers(
    n: int = 5,
    metric: str = "sif_density_pct"
):
    _, _, _, barrier = load_data()

    allowed_metrics = [
        "sif_density_pct",
        "sif_reports",
        "total_reports"
    ]

    if metric not in allowed_metrics:
        metric = "sif_density_pct"

    result = (
        barrier
        .sort_values(metric, ascending=False)
        .head(n)
    )

    return result.to_dict(orient="records")


def get_top_lsr(
    n: int = 5,
    metric: str = "sif_density_pct"
):
    _, lsr, _, _ = load_data()

    allowed_metrics = [
        "sif_density_pct",
        "sif_reports",
        "total_reports"
    ]

    if metric not in allowed_metrics:
        metric = "sif_density_pct"

    result = (
        lsr
        .sort_values(metric, ascending=False)
        .head(n)
    )

    return result.to_dict(orient="records")


def get_sif_statistics():
    df = load_master_data()

    total_reports = len(df)

    result = {
        "total_reports": int(total_reports)
    }

    if "sif_prediction" in df.columns:
        counts = (
            df["sif_prediction"]
            .value_counts(dropna=False)
            .to_dict()
        )

        sif_count = int(counts.get("SIF Potential", 0))
        non_sif_count = int(counts.get("Non-SIF Potential", 0))

        result["sif_potential"] = sif_count
        result["non_sif_potential"] = non_sif_count

        if total_reports > 0:
            result["sif_percentage"] = round(
                (sif_count / total_reports) * 100,
                2
            )

    if "sif_probability" in df.columns:
        probabilities = pd.to_numeric(
            df["sif_probability"],
            errors="coerce"
        )

        result["average_sif_probability"] = round(
            float(probabilities.mean()),
            4
        )

    return result

def dataset_statistics():
    return get_sif_statistics()


def top_precursors(n: int, metric: str):
    return get_top_precursors(n, metric)


def top_activities(n: int, metric: str):
    return get_top_activities(n, metric)


def top_barriers(n: int, metric: str):
    return get_top_barriers(n, metric)


def top_life_saving_rules(n: int, metric: str):
    return get_top_lsr(n, metric)


def find_incidents(search_term: str, limit: int):
    return search_incidents(
        search_term,
        limit
    )


def get_incident_by_id(report_id: str):
    df = load_master_data()

    # Find the report ID column
    id_column = None

    for column in [
        "report_id",
        "Report ID",
        "reportid",
        "id",
        "ID"
    ]:
        if column in df.columns:
            id_column = column
            break

    if id_column is None:
        return None

    result = df[
        df[id_column]
        .astype(str)
        .str.strip()
        .str.upper()
        == report_id.strip().upper()
    ]

    if result.empty:
        return None

    record = result.iloc[0].to_dict()

    # Make the record safe to work with
    cleaned = {}

    for key, value in record.items():

        if pd.isna(value):
            cleaned[key] = None

        else:
            cleaned[key] = value

    return cleaned