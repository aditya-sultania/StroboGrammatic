import numpy as np
import pandas as pd

SEVERE_EVENTS = (
    "FATAL", "DEATH", "ELECTROCUT", "AMPUT", "CRUSH", "STRUCK BY",
    "CAUGHT", "EXPLOS", "FIRE", "COLLAPSE", "FALL FROM", "DROWNING"
)


def state_ranking(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw[raw["State"].notna() & (raw["State"] != "")].copy()
    d["severe_event"] = d["EventTitle"].fillna("").astype(str).str.upper().apply(
        lambda x: int(any(k in x for k in SEVERE_EVENTS))
    )
    out = d.groupby("State", as_index=False).agg(
        incidents=("ID", "count"),
        employers=("Employer", "nunique"),
        hospitalized=("Hospitalized", "sum"),
        amputations=("Amputation", "sum"),
        loss_of_eye=("Loss of Eye", "sum"),
        severe_events=("severe_event", "sum"),
    )
    out["severity_points"] = (
        out["hospitalized"] * 2
        + out["amputations"] * 8
        + out["loss_of_eye"] * 5
        + out["severe_events"] * 3
    )
    out["severity_rate_per_100"] = (out["severity_points"] / out["incidents"].clip(lower=1) * 100).round(2)
    # Transparent ranking: 65% incident burden + 35% severity rate percentile.
    burden_pct = out["incidents"].rank(pct=True)
    sev_pct = out["severity_rate_per_100"].rank(pct=True)
    out["risk_index"] = ((0.65 * burden_pct + 0.35 * sev_pct) * 100).round(1)
    out["risk_band"] = pd.cut(
        out["risk_index"], [-np.inf, 33, 66, np.inf], labels=["Lower", "Moderate", "Elevated"]
    ).astype(str)
    return out.sort_values(["risk_index", "incidents"], ascending=False).reset_index(drop=True)


def top_hazards(raw: pd.DataFrame, state=None, n=10):
    d = raw.copy()
    if state and state != "All states":
        d = d[d["State"] == state]
    d["hazard"] = d["NatureTitle"].fillna(d["EventTitle"]).fillna("Unknown").astype(str).str.strip()
    return d["hazard"].replace("", "Unknown").value_counts().head(n).rename_axis("hazard").reset_index(name="incidents")


def state_hazard_table(raw: pd.DataFrame, state, n=10):
    return top_hazards(raw, state, n)


def trend(raw: pd.DataFrame, state=None):
    d = raw.dropna(subset=["EventDate"]).copy()
    if state and state != "All states":
        d = d[d["State"] == state]
    d["month"] = d["EventDate"].dt.to_period("M").dt.to_timestamp()
    return d.groupby("month", as_index=False).size().rename(columns={"size": "incidents"})


def master_kpis(master):
    total = len(master)
    sif = int((master["sif_prediction"].astype(str).str.lower() == "sif potential").sum())
    review = int((master.get("hse_review_status", pd.Series(dtype=str)).astype(str) == "Review Suggested").sum()) if "hse_review_status" in master else 0
    return {
        "total": total,
        "sif": sif,
        "sif_rate": (sif / total * 100) if total else 0,
        "review": review,
        "mean_probability": float(master["sif_probability"].mean()) if total else 0,
    }


def filtered_master(master, activity="All", precursor="All", min_prob=0.0):
    d = master.copy()
    if activity != "All": d = d[d["activity_group"] == activity]
    if precursor != "All": d = d[d["precursor"] == precursor]
    d = d[d["sif_probability"] >= min_prob]
    return d.sort_values(["priority_score", "sif_probability"], ascending=False)


def recommendations_for_hazard(hazard: str):
    h = hazard.upper()
    rules = []
    if any(x in h for x in ["FALL", "LADDER", "ROOF", "ELEVAT"]):
        rules += ["Working at Height", "Use approved fall protection", "Verify anchor points before work"]
    if any(x in h for x in ["VEHICLE", "FORKLIFT", "STRUCK", "MOBILE"]):
        rules += ["Driving / Mobile Equipment", "Separate pedestrians and vehicles", "Verify exclusion zones and spotter controls"]
    if any(x in h for x in ["ELECTRIC", "ELECTROCUT"]):
        rules += ["Energy Isolation", "Lock out and verify zero energy before work", "Use competent-person electrical controls"]
    if any(x in h for x in ["FIRE", "BURN", "WELD", "HOT"]):
        rules += ["Hot Work", "Control ignition sources", "Confirm gas testing/fire watch where required"]
    if any(x in h for x in ["CRUSH", "CAUGHT", "MACHINE", "AMPUT"]):
        rules += ["Line of Fire", "Verify guarding and isolation", "Keep body parts out of pinch/crush zones"]
    if not rules:
        rules = ["Stop and assess the task", "Verify critical controls before starting", "Escalate recurring unsafe conditions to HSE"]
    return list(dict.fromkeys(rules))
