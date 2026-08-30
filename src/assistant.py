import pandas as pd
from .analytics import recommendations_for_hazard


def answer(question: str, raw: pd.DataFrame, master: pd.DataFrame) -> str:
    q = question.lower().strip()
    if not q:
        return "Ask about a state, hazard, SIF potential, precursor, trend, or recommended control."

    # State-specific questions.
    states = [s for s in raw["State"].dropna().unique() if len(str(s)) > 2]
    state = next((s for s in states if str(s).lower() in q), None)
    if state:
        d = raw[raw["State"] == state]
        hazards = d["NatureTitle"].fillna(d["EventTitle"]).fillna("Unknown").astype(str).value_counts().head(5)
        top = ", ".join([f"{k} ({v})" for k, v in hazards.items()])
        return (
            f"{state.title()} has {len(d):,} incident records in the supplied OSHA dataset. "
            f"The leading observed hazards/events are {top}. "
            "This is an observed incident-burden view, not a claim that the state is inherently unsafe. "
            "Use the HSE drill-down to inspect the underlying records and controls."
        )

    if "sif" in q or "precursor" in q:
        sif = master[master["sif_prediction"].astype(str).str.lower() == "sif potential"]
        if len(sif):
            top = sif["precursor"].value_counts().head(5)
            txt = ", ".join([f"{k} ({v})" for k, v in top.items()])
            return f"The master SIF dataset contains {len(sif):,} SIF-potential records. Top precursors are {txt}. Prioritize high probability/high priority records for HSE review."

    if "hazard" in q or "common" in q:
        hz = raw["NatureTitle"].fillna(raw["EventTitle"]).fillna("Unknown").astype(str).value_counts().head(7)
        return "Across the supplied OSHA dataset, the most frequent observed hazard/event labels are: " + "; ".join(f"{k} ({v})" for k, v in hz.items()) + "."

    if any(x in q for x in ["prevent", "avoid", "recommend", "control"]):
        hz = raw["NatureTitle"].fillna(raw["EventTitle"]).fillna("Unknown").astype(str).value_counts().index[0]
        return "Recommended control themes for the leading observed hazard are: " + "; ".join(recommendations_for_hazard(hz)) + "."

    return (
        "I can analyze the supplied OSHA incident data and SIF master data. Try: "
        "'Which state has the highest incident burden?', 'What are the top SIF precursors?', "
        "'What hazards are common in Texas?', or 'How can we prevent the top hazard?'."
    )
