import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

from chatbot.assistant import answer_question


# ============================================================
# PATHS / MODELS
# ============================================================

ROOT = Path(__file__).parent

SIF_MODEL = joblib.load(
    ROOT / "SIF_Model_v4_DomainAware.joblib"
)

LSR_MODEL = joblib.load(
    ROOT / "IOGP_Life_Saving_Rule_Classifier_v2.joblib"
)


# ============================================================
# SAFETY / CLASSIFICATION RULES
# ============================================================

HIGH_RISK = {
    "Confined Space":
        r"confined space|asphyxi|hydrogen sulfide|\bh2s\b|oxygen deficient|engulf",

    "Energy / Isolation":
        r"electrocution|electrical shock|arc flash|high voltage|energized|lock[- ]out|tag[- ]out|stored energy",

    "Line of Fire":
        r"caught between|pinned between|crushed between|suspended load|line of fire|struck by",

    "Working at Height":
        r"fall(?:ed|ing)? from|fall from height|roof|scaffold|tower|derrick",

    "Hot Work / Fire":
        r"explosion|flash fire|fireball|uncontrolled release|flammable gas",

    "Excavation":
        r"trench collapse|cave[- ]in|buried in|excavation collapse",

    "Pressure":
        r"high pressure|pressure release|pressurized|rupture",

    "Drowning / Water":
        r"drown(?:ed|ing)?|man overboard|submerged"
}


LSR = {
    "Energy Isolation":
        r"energized|high voltage|electrocution|electrical shock|arc flash|lock[- ]out|tag[- ]out|stored energy|isolation",

    "Hot Work":
        r"hot work|welding|cutting|grinding|flammable|explosion|fire",

    "Confined Space":
        r"confined space|asphyxi|hydrogen sulfide|\bh2s\b|oxygen deficient|engulf",

    "Line of Fire":
        r"line of fire|struck by|caught between|pinned between|crushed between|suspended load|falling object",

    "Working at Height":
        r"fall|roof|scaffold|tower|derrick|elevated platform",

    "Driving / Mobile Equipment":
        r"vehicle|forklift|truck|driving|excavat|mobile equipment",

    "Lifting Operations":
        r"lifting|load|rigging|hoist|crane|suspended load",

    "Excavation":
        r"excavat|trench|cave[- ]in|buried|collapse"
}


PREC = {
    "Energy / isolation":
        r"energized|high voltage|electrocution|electrical shock|arc flash|lock[- ]out|tag[- ]out|stored energy|power cable",

    "Confined space":
        r"confined space|asphyxi|hydrogen sulfide|\bh2s\b|oxygen deficient|engulf",

    "Line of fire":
        r"struck by|caught between|pinned between|crushed between|suspended load|falling object|line of fire",

    "Working at height":
        r"fall(?:ed|ing)? from|fall(?:ed|ing)? off|roof|scaffold|tower|derrick|elevated platform",

    "Hot work / fire":
        r"hot work|welding|cutting|grinding|explosion|flash fire|flammable",

    "Excavation":
        r"excavat|trench|cave[- ]in|buried|collapse",

    "Vehicle / mobile equipment":
        r"forklift|excavator|vehicle|run over|backed over|mobile equipment",

    "Machinery / guarding":
        r"caught in|entangled|unguarded|machine guard|interlock|bypass"
}


ACT = {
    "Maintenance":
        r"maintenance|repair|cleaning|servicing",

    "Material handling / lifting":
        r"lifting|load|rigging|hoist|material handling|crane",

    "Working at height":
        r"fall|roof|scaffold|tower|elevated",

    "Mobile equipment / vehicles":
        r"vehicle|forklift|truck|excavat|transport",

    "Machinery operation":
        r"machin|equipment|caught in|entangled",

    "Excavation":
        r"excavat|trench",

    "Welding / hot work":
        r"weld|cutting|grinding|hot work",

    "Electrical work":
        r"electric|electrical|power cable|energized"
}


BARR = {
    "Isolation failure":
        [
            "isolation",
            "isolat",
            "lock-out",
            "lockout",
            "tag-out",
            "tagout",
            "energized",
            "stored energy"
        ],

    "Permit / authorization failure":
        [
            "permit",
            "authorization",
            "work authorization"
        ],

    "Planning / risk assessment":
        [
            "planning",
            "risk assessment",
            "hazard assessment",
            "jsa",
            "job safety"
        ],

    "Communication failure":
        [
            "communication",
            "communicat",
            "handover",
            "coordination"
        ],

    "Supervision failure":
        [
            "supervision",
            "supervisor",
            "oversight"
        ],

    "Procedure failure":
        [
            "procedure",
            "procedural",
            "instruction",
            "standard"
        ],

    "Training / competency":
        [
            "training",
            "competenc",
            "qualified",
            "qualification"
        ],

    "Inspection / maintenance":
        [
            "inspection",
            "inspect",
            "maintenance",
            "defect",
            "equipment condition"
        ],

    "Safety control bypass":
        [
            "bypass",
            "interlock",
            "guard removed",
            "safety control"
        ]
}


# ============================================================
# HELPERS
# ============================================================

def hits(text, dictionary):
    """
    Return matching categories for a piece of text.
    """

    return [
        key
        for key, pattern in dictionary.items()
        if re.search(pattern, text, re.I)
    ]


def explain(text):
    """
    Explainable features from the trained linear model.
    """

    features = SIF_MODEL.named_steps["features"]
    clf = SIF_MODEL.named_steps["clf"]

    X = features.transform([text]).toarray()[0]

    names = features.get_feature_names_out()

    contributions = X * clf.coef_[0]

    positive = sorted(
        [
            (names[i], float(contributions[i]))
            for i in range(len(contributions))
            if contributions[i] > 0
        ],
        key=lambda x: x[1],
        reverse=True
    )[:8]

    negative = sorted(
        [
            (names[i], float(contributions[i]))
            for i in range(len(contributions))
            if contributions[i] < 0
        ],
        key=lambda x: x[1]
    )[:8]

    return positive, negative


def simple_explanation(
    text,
    probability,
    decision,
    high_risk,
    precursors,
    activities,
    barriers,
    lsr
):
    """
    Human-readable explanation for HSE users.

    This is an interpretation of model/rule outputs,
    not a causal explanation of the incident.
    """

    hazard_phrases = []

    if high_risk:
        hazard_phrases.extend(high_risk)

    if precursors:
        hazard_phrases.extend(precursors)

    if decision == "SIF Potential":

        if high_risk:

            why = (
                "The report contains a high-risk mechanism associated "
                "with serious or fatal outcomes: "
                + ", ".join(high_risk)
                + ". That mechanism is why the case is being prioritized "
                  "for SIF review."
            )

        elif precursors:

            why = (
                "The report contains precursor signals such as "
                + ", ".join(precursors[:3])
                + ". These signals increase the model's estimated "
                  "likelihood of SIF potential."
            )

        else:

            why = (
                "The NLP model found language patterns associated with "
                "SIF-potential reports. Because the evidence is not a "
                "direct causal assessment, an HSE review is recommended."
            )

    else:

        why = (
            "The report does not contain strong signals of the "
            "high-energy SIF mechanisms used by the screening system. "
            "The model therefore estimates a lower SIF probability. "
            "This does not mean the event is unimportant; it means it "
            "is lower priority for SIF screening."
        )

    findings = []

    if activities:
        findings.append(
            "Activity: " + ", ".join(activities[:2])
        )

    if precursors:
        findings.append(
            "Precursor: " + ", ".join(precursors[:3])
        )

    if barriers:
        findings.append(
            "Barrier theme: " + ", ".join(barriers[:2])
        )

    if lsr:
        findings.append(
            "Life-Saving Rule: " + lsr[0][0]
        )

    if decision == "SIF Potential" or probability >= 0.30:

        action = (
            "Recommended HSE action: review the report, verify the "
            "hazard and failed barriers, and confirm or correct the "
            "SIF/LSR classification."
        )

    else:

        action = (
            "Recommended HSE action: retain normal follow-up. Escalate "
            "if additional context shows a high-energy exposure or "
            "another SIF mechanism that is not present in the text."
        )

    return why, findings, action


def review_level(probability, high_risk):
    """
    Determine HSE review priority.
    """

    if high_risk or probability >= 0.70:
        return "High Priority"

    if probability >= 0.30:
        return "HSE Review"

    return "Normal"


def analyze(text):
    """
    Run the SIF model, LSR model, rule-based safety backstops,
    and explainability logic.
    """

    probability = float(
        SIF_MODEL.predict_proba([text])[0, 1]
    )

    high_risk = hits(
        text,
        HIGH_RISK
    )

    prediction = (
        "SIF Potential"
        if probability >= 0.5 or high_risk
        else "Non-SIF Potential"
    )

    lsr_probabilities = LSR_MODEL.predict_proba([text])[0]

    indices = lsr_probabilities.argsort()[::-1][:3]

    lsr = [
        (
            str(LSR_MODEL.classes_[i]),
            float(lsr_probabilities[i])
        )
        for i in indices
    ]

    precursors = hits(
        text,
        PREC
    )

    activities = hits(
        text,
        ACT
    )

    lsr_matches = hits(
        text,
        LSR
    )

    barrier_patterns = {
        key: "|".join(
            map(re.escape, values)
        )
        for key, values in BARR.items()
    }

    barriers = hits(
        text,
        barrier_patterns
    )

    xai = explain(text)

    return (
        probability,
        prediction,
        high_risk,
        precursors,
        activities,
        lsr_matches,
        lsr,
        barriers,
        xai
    )


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="OIL SIF Intelligence",
    layout="wide"
)

st.title(
    "🛡️ OIL SIF Precursor Intelligence"
)

st.caption(
    "AI/NLP prototype • SIF screening • IOGP Life-Saving Rules • "
    "XAI • HSE prioritization"
)


# ============================================================
# LOAD DASHBOARD DATA
# ============================================================

master = pd.read_csv(
    ROOT / "SIF_Dashboard_Master_v4.csv",
    low_memory=False
)

activity = pd.read_csv(
    ROOT / "SIF_Activity_Ranking_v4.csv"
)

lsr = pd.read_csv(
    ROOT / "SIF_LSR_Ranking_v4.csv"
)

pre = pd.read_csv(
    ROOT / "SIF_Precursor_Ranking_v4.csv"
)

bar = pd.read_csv(
    ROOT / "SIF_Barrier_Ranking_v4.csv"
)

queue = pd.read_csv(
    ROOT / "SIF_HSE_Review_Queue_v2.csv",
    low_memory=False
)


# ============================================================
# DASHBOARD TABS
# ============================================================

tabs = st.tabs(
    [
        "Overview",
        "SIF Risk",
        "Site / Activity Ranking",
        "Life-Saving Rules",
        "Precursors",
        "Barrier Failures",
        "XAI / Analyze",
        "HSE Review Queue"
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with tabs[0]:

    st.info(
        "Prototype uses public OSHA data. OIL site ranking requires "
        "OIL HSSE UA/UC, near-miss and incident data."
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "Reports",
        f"{len(master):,}"
    )

    a = a

    b.metric(
        "SIF Potential",
        f"{(master.sif_prediction == 'SIF Potential').sum():,}"
    )

    c.metric(
        "SIF Density",
        f"{100 * (master.sif_prediction == 'SIF Potential').mean():.2f}%"
    )

    d.metric(
        "HSE Review Queue",
        f"{len(queue):,}"
    )

    st.subheader("Risk distribution")

    risk = (
        master.sif_prediction
        .value_counts()
        .reset_index()
    )

    risk.columns = [
        "Prediction",
        "Reports"
    ]

    st.plotly_chart(
        px.pie(
            risk,
            names="Prediction",
            values="Reports",
            hole=.45
        ),
        use_container_width=True
    )


# ============================================================
# SIF RISK
# ============================================================

with tabs[1]:

    st.subheader("SIF Risk")

    st.plotly_chart(
        px.histogram(
            master,
            x="sif_probability",
            nbins=30,
            title="SIF probability distribution"
        ),
        use_container_width=True
    )

    st.dataframe(
        master[
            [
                "report_id",
                "sif_probability",
                "sif_prediction",
                "activity_group",
                "precursor"
            ]
        ]
        .sort_values(
            "sif_probability",
            ascending=False
        )
        .head(1000),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ACTIVITY
# ============================================================

with tabs[2]:

    st.subheader("Activity ranking")

    st.plotly_chart(
        px.bar(
            activity,
            x="sif_density_pct",
            y="activity_group",
            orientation="h",
            title="SIF-precursor density by activity"
        ),
        use_container_width=True
    )

    st.caption(
        "Site ranking is disabled for public data because the "
        "dataset does not contain OIL's site hierarchy."
    )

    st.dataframe(
        activity,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# LIFE-SAVING RULES
# ============================================================

with tabs[3]:

    st.subheader(
        "IOGP Life-Saving Rules"
    )

    st.plotly_chart(
        px.bar(
            lsr,
            x="sif_reports",
            y="life_saving_rule",
            orientation="h",
            title="SIF-potential reports by Life-Saving Rule"
        ),
        use_container_width=True
    )

    st.dataframe(
        lsr,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PRECURSORS
# ============================================================

with tabs[4]:

    st.subheader(
        "Recurring precursor patterns"
    )

    st.plotly_chart(
        px.bar(
            pre,
            x="sif_reports",
            y="precursor",
            orientation="h",
            title="SIF-potential reports by precursor"
        ),
        use_container_width=True
    )

    st.dataframe(
        pre,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# BARRIER FAILURES
# ============================================================

with tabs[5]:

    st.subheader(
        "Barrier failures"
    )

    st.plotly_chart(
        px.bar(
            bar,
            x="sif_reports",
            y="barrier_theme",
            orientation="h",
            title="SIF-potential reports by barrier-failure theme"
        ),
        use_container_width=True
    )

    st.dataframe(
        bar,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# XAI
# ============================================================

with tabs[6]:

    st.subheader(
        "Explainable AI — analyze one report"
    )

    sample = st.selectbox(
        "Example",
        [
            "Custom",
            "Worker entered a confined space without gas testing. "
            "Oxygen level was not checked.",

            "During excavation, the excavator bucket contacted an "
            "underground power cable.",

            "A suspended load moved unexpectedly and nearly struck "
            "a worker.",

            "Employee slipped on a wet office floor and sustained "
            "a minor bruise."
        ]
    )

    txt = st.text_area(
        "Safety report",
        "" if sample == "Custom" else sample,
        height=150
    )

    if st.button(
        "Analyze",
        type="primary"
    ):

        if txt.strip():

            (
                probability,
                prediction,
                high_risk,
                precursors,
                activities,
                lsr_matches,
                lsr_predictions,
                barriers,
                xai
            ) = analyze(txt)

            positive_features, negative_features = xai

            review = review_level(
                probability,
                high_risk
            )

            why, findings, action = simple_explanation(
                txt,
                probability,
                prediction,
                high_risk,
                precursors,
                activities,
                barriers,
                lsr_predictions
            )

            a, b, c = st.columns(3)

            a.metric(
                "SIF probability",
                f"{probability * 100:.1f}%"
            )

            b.metric(
                "Decision",
                prediction
            )

            c.metric(
                "Review",
                review
            )

            st.subheader(
                "🧠 Simple explanation"
            )

            st.info(why)

            if findings:

                st.markdown(
                    "**What the system found:**"
                )

                for item in findings:
                    st.write(
                        "• " + item
                    )

            st.success(action)

            if high_risk:

                st.warning(
                    "Safety backstop triggered: "
                    + ", ".join(high_risk)
                )

            with st.expander(
                "🔬 Technical XAI — model feature contributions"
            ):

                st.caption(
                    "These are model-feature contributions, not proof "
                    "that a word caused the incident."
                )

                x, y = st.columns(2)

                with x:

                    st.markdown(
                        "**Features pushing toward SIF**"
                    )

                    for key, value in positive_features:
                        st.write(
                            f"• `{key}` — {value:.4f}"
                        )

                with y:

                    st.markdown(
                        "**Features pushing toward Non-SIF**"
                    )

                    for key, value in negative_features:
                        st.write(
                            f"• `{key}` — {value:.4f}"
                        )

            st.subheader(
                "Life-Saving Rule"
            )

            st.write(
                "**"
                + lsr_predictions[0][0]
                + "** — "
                + f"{lsr_predictions[0][1] * 100:.1f}%"
            )

            st.caption(
                "Top 3: "
                + " • ".join(
                    f"{key} {value * 100:.1f}%"
                    for key, value in lsr_predictions
                )
            )

            x, y = st.columns(2)

            with x:

                st.write(
                    "**Precursors:** "
                    + (
                        ", ".join(precursors)
                        if precursors
                        else "None detected"
                    )
                )

                st.write(
                    "**Activity:** "
                    + (
                        ", ".join(activities)
                        if activities
                        else "None detected"
                    )
                )

            with y:

                st.write(
                    "**LSR keyword backstop:** "
                    + (
                        ", ".join(lsr_matches)
                        if lsr_matches
                        else "None"
                    )
                )

                st.write(
                    "**Barrier themes:** "
                    + (
                        ", ".join(barriers)
                        if barriers
                        else "None detected"
                    )
                )

        else:

            st.warning(
                "Enter a report."
            )


# ============================================================
# HSE REVIEW QUEUE
# ============================================================

with tabs[7]:

    st.subheader(
        "HSE Review Queue"
    )

    st.write(
        "Medium-confidence reports are candidates for human review; "
        "high-probability reports are prioritized."
    )

    if "review_priority" not in queue.columns:

        queue["review_priority"] = pd.cut(
            queue["sif_probability"],
            bins=[
                -0.001,
                0.30,
                0.70,
                1.001
            ],
            labels=[
                "Normal",
                "HSE Review",
                "High Priority"
            ]
        )

    if "hse_review_status" not in queue.columns:

        queue["hse_review_status"] = (
            queue["review_priority"]
            .astype(str)
            .map(
                {
                    "Normal": "Not Queued",
                    "HSE Review": "Review Suggested",
                    "High Priority": "Pending HSE Review"
                }
            )
        )

    required_cols = [
        "report_id",
        "sif_probability",
        "sif_prediction",
        "review_priority",
        "hse_review_status",
        "activity_group",
        "precursor",
        "life_saving_rule"
    ]

    available_cols = [
        column
        for column in required_cols
        if column in queue.columns
    ]

    st.dataframe(
        queue[available_cols].head(5000),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# SIF ASSISTANT
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "💬 SIF Assistant"
)

st.sidebar.info(
    "Ask questions about SIF incidents, "
    "precursors, barriers, activities, "
    "Life-Saving Rules and risk."
)


# ------------------------------------------------------------
# Gemini chat session
# ------------------------------------------------------------

if "gemini_chat" not in st.session_state:
    st.session_state.gemini_chat = None


# ------------------------------------------------------------
# Visible conversation history
# ------------------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ------------------------------------------------------------
# Clear conversation
# ------------------------------------------------------------

if st.sidebar.button(
    "🗑️ Clear Chat"
):

    st.session_state.chat_history = []
    st.session_state.gemini_chat = None

    st.rerun()


# ------------------------------------------------------------
# Display conversation
# ------------------------------------------------------------

for role, message in st.session_state.chat_history:

    with st.chat_message(role):
        st.markdown(message)


# ------------------------------------------------------------
# Chat input
# ------------------------------------------------------------

user_question = st.chat_input(
    "Ask anything about SIF incidents, risks, precursors, barriers..."
)


if user_question:

    # --------------------------------------------------------
    # User message
    # --------------------------------------------------------

    st.session_state.chat_history.append(
        (
            "user",
            user_question
        )
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    # --------------------------------------------------------
    # Gemini response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Analyzing..."
        ):

            answer, new_chat = answer_question(
                user_question,
                st.session_state.gemini_chat
            )

            st.session_state.gemini_chat = new_chat

            st.markdown(answer)

    # --------------------------------------------------------
    # Store assistant response
    # --------------------------------------------------------

    st.session_state.chat_history.append(
        (
            "assistant",
            answer
        )
    )