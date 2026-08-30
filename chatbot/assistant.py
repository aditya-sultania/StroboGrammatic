import re

from chatbot.data_tools import (
    dataset_statistics,
    top_precursors,
    top_activities,
    top_barriers,
    top_life_saving_rules,
    find_incidents,
    get_incident_by_id,
)


# ============================================================
# HELPERS
# ============================================================

def _get_number(question, default=5):
    """
    Extract a requested number from the question.
    Examples:
        "top 10 activities" -> 10
        "top five precursors" -> 5
    """

    match = re.search(r"\btop\s+(\d+)\b", question.lower())

    if match:
        return max(1, min(int(match.group(1)), 20))

    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }

    for word, number in words.items():
        if re.search(rf"\btop\s+{word}\b", question.lower()):
            return number

    return default


def _get_metric(question):
    """
    Decide which dataset metric the user is asking about.
    """

    q = question.lower()

    if (
        "sif density" in q
        or "density" in q
        or "percentage" in q
        or "percent" in q
    ):
        return "sif_density_pct"

    if (
        "sif reports" in q
        or "number of sif" in q
        or "most sif" in q
        or "highest sif" in q
    ):
        return "sif_reports"

    if (
        "total reports" in q
        or "most reports" in q
        or "highest number of reports" in q
    ):
        return "total_reports"

    # Default ranking
    return "sif_density_pct"


def _format_percentage(value):
    try:
        return f"{float(value):.2f}%"
    except Exception:
        return str(value)


def _format_ranking(records, name_column, metric):
    """
    Convert ranking records into a readable chatbot answer.
    """

    if not records:
        return "I couldn't find any matching records in the OIL SIF dataset."

    metric_names = {
        "sif_density_pct": "SIF density",
        "sif_reports": "SIF reports",
        "total_reports": "total reports",
    }

    metric_name = metric_names.get(metric, metric)

    lines = [
        f"Here are the top {len(records)} results by {metric_name}:",
        ""
    ]

    for i, record in enumerate(records, 1):

        name = record.get(name_column, "Unknown")

        if metric == "sif_density_pct":
            value = _format_percentage(
                record.get("sif_density_pct", 0)
            )

        else:
            value = str(
                record.get(metric, 0)
            )

        lines.append(
            f"{i}. **{name}** — {value}"
        )

    return "\n".join(lines)


# ============================================================
# DATASET STATISTICS
# ============================================================

def _answer_statistics():
    data = dataset_statistics()

    if not data:
        return "I couldn't retrieve the overall dataset statistics."

    total = data.get("total_reports")
    sif_counts = data.get("sif_prediction_counts", {})
    avg_probability = data.get("average_sif_probability")

    sif_potential = sif_counts.get(
        "SIF Potential",
        0
    )

    non_sif = sif_counts.get(
        "Non-SIF Potential",
        0
    )

    if total:
        sif_percentage = (
            sif_potential / total
        ) * 100
    else:
        sif_percentage = 0

    return f"""
### OIL SIF Dataset Overview

- **Total reports:** {total:,}
- **SIF Potential:** {sif_potential:,}
- **Non-SIF Potential:** {non_sif:,}
- **SIF Potential percentage:** {sif_percentage:.2f}%
- **Average SIF probability:** {float(avg_probability) * 100:.2f}%

These figures come directly from the OIL SIF dataset and its associated SIF prediction results.
""".strip()


# ============================================================
# INCIDENT SEARCH
# ============================================================

def _answer_incidents(question):

    q = question.lower().strip()

    # --------------------------------------------------------
    # Remove search-command language
    # --------------------------------------------------------

    search_term = q

    phrases_to_remove = [
        "find incidents involving",
        "find incidents related to",
        "find incidents about",

        "show incidents involving",
        "show incidents related to",
        "show incidents about",

        "search incidents involving",
        "search incidents related to",
        "search incidents about",

        "find incidents",
        "show incidents",
        "search incidents",

        "find incident",
        "show incident",
        "search incident",

        "find",
        "show me",
        "show",
        "search for",
        "search",

        "incidents",
        "incident",
    ]

    for phrase in phrases_to_remove:
        search_term = search_term.replace(phrase, "")

    # Clean common connecting words
    search_term = re.sub(
        r"^\s*(involving|related to|about|with|for)\s+",
        "",
        search_term
    )

    search_term = search_term.strip()

    if not search_term:
        return (
            "Please specify what type of incident you want me to search "
            "for, for example **forklift**, **lifting**, or "
            "**confined space**."
        )

    records = find_incidents(
        search_term=search_term,
        limit=10
    )

    if not records:
        return (
            f"I couldn't find incidents matching **{search_term}** "
            "in the OIL SIF dataset."
        )

    lines = [
        f"### Incidents involving {search_term}",
        "",
        f"I found **{len(records)} matching records** in the dataset.",
        ""
    ]

    for i, record in enumerate(records, 1):

        report_id = (
            record.get("report_id")
            or record.get("Report ID")
            or record.get("id")
            or "Unknown"
        )

        description = (
            record.get("description")
            or record.get("incident_description")
            or record.get("report_text")
            or record.get("incident")
            or "No description available."
        )

        prediction = (
            record.get("sif_prediction")
            or record.get("SIF Prediction")
            or record.get("prediction")
            or "Not available"
        )

        probability = (
            record.get("sif_probability")
            or record.get("SIF Probability")
        )

        lines.append(
            f"**{i}. {report_id}**"
        )

        lines.append(
            f"- {description}"
        )

        lines.append(
            f"- **SIF classification:** {prediction}"
        )

        if probability is not None:
            try:
                probability = float(probability)

                # Handle either 0-1 or 0-100 representation
                if probability <= 1:
                    probability *= 100

                lines.append(
                    f"- **SIF probability:** {probability:.1f}%"
                )

            except Exception:
                pass

        lines.append("")

    lines.append(
        "*SIF classifications and probabilities are machine-learning "
        "outputs intended to support safety review and do not replace "
        "formal HSE assessment.*"
    )

    return "\n".join(lines)


# ============================================================
# MAIN LOCAL QUERY ENGINE
# ============================================================

def _answer_incident_lookup(question):

    match = re.search(
        r"\b(?:ITA[_-]?\d+)\b",
        question,
        re.IGNORECASE
    )

    if not match:
        return None

    report_id = match.group(0).upper().replace("-", "_")

    record = get_incident_by_id(report_id)

    if not record:
        return (
            f"I couldn't find report **{report_id}** "
            "in the OIL SIF dataset."
        )

    probability = (
        record.get("sif_probability")
        or record.get("SIF Probability")
        or record.get("sif_probability_pct")
    )

    prediction = (
        record.get("sif_prediction")
        or record.get("SIF Prediction")
        or record.get("prediction")
    )

    lines = [
        f"### Report {report_id}",
        ""
    ]

    if probability is not None:
        try:
            probability = float(probability)

            if probability <= 1:
                probability *= 100

            lines.append(
                f"- **SIF probability:** {probability:.1f}%"
            )

        except (ValueError, TypeError):
            lines.append(
                f"- **SIF probability:** {probability}"
            )

    if prediction:
        lines.append(
            f"- **SIF classification:** {prediction}"
        )

    activity = (
        record.get("activity_group")
        or record.get("Activity Group")
    )

    if activity:
        lines.append(
            f"- **Activity:** {activity}"
        )

    precursor = (
        record.get("precursor")
        or record.get("Precursor")
    )

    if precursor:
        lines.append(
            f"- **Precursor:** {precursor}"
        )

    lsr = (
        record.get("life_saving_rule")
        or record.get("Life Saving Rule")
        or record.get("life_saving_rules")
    )

    if lsr:
        lines.append(
            f"- **Life-Saving Rule:** {lsr}"
        )

    description = (
        record.get("description")
        or record.get("incident_description")
        or record.get("report_text")
        or record.get("incident")
    )

    if description:
        lines.extend([
            "",
            f"**Incident:** {description}"
        ])

    lines.extend([
        "",
        "*SIF probability and classification are machine-learning "
        "outputs intended to support safety review and do not replace "
        "formal HSE assessment.*"
    ])

    return "\n".join(lines)

def answer_question(question):

    if not question or not question.strip():
        return "Please enter a question."

    q = question.lower().strip()

        # --------------------------------------------------------
    # INDIVIDUAL INCIDENT LOOKUP
    # --------------------------------------------------------

    incident_lookup = _answer_incident_lookup(question)

    if incident_lookup:
        return incident_lookup

    # --------------------------------------------------------
    # GENERAL SIF KNOWLEDGE
    # --------------------------------------------------------

    if (
        "what is sif" in q
        or "what are sif" in q
        or "explain sif" in q
        or "meaning of sif" in q
    ):
        return """
### What is a Safety Instrumented Function?

A **Safety Instrumented Function (SIF)** is an automated safety system designed to detect a specific hazardous condition and move a process to a safe state.

A SIF consists of three main parts:

1. **Sensor** — detects a dangerous condition such as excessive pressure or temperature.
2. **Logic solver** — evaluates the sensor signal and determines whether the safety action is required.
3. **Final element** — performs the physical action, such as closing a valve or shutting down equipment.

In simple terms:

**Sense → Decide → Act**

A SIF is an independent layer of protection intended to reduce the likelihood or consequences of a hazardous event.
""".strip()

    # --------------------------------------------------------
    # DATASET STATISTICS
    # --------------------------------------------------------

    if (
        "how many reports" in q
        or "total reports" in q
        or "dataset size" in q
        or "how large is the dataset" in q
        or "dataset contain" in q
        or "overall statistics" in q
        or "dataset statistics" in q
        or "overall data" in q
    ):
        return _answer_statistics()

    # --------------------------------------------------------
    # INCIDENT SEARCH
    # --------------------------------------------------------

    incident_words = [
    "find incidents",
    "find incident",
    "find forklift incidents",
    "find lifting incidents",
    "find confined space incidents",
    "show incidents",
    "show incident",
    "show forklift incidents",
    "show lifting incidents",
    "show confined space incidents",
    "search incidents",
    "search incident",
    "search for incidents",
    "search for incident",
    "incidents involving",
    "incident involving",
    "incidents related to",
    "incident related to",
    "incidents about",
    "incident about",
]

    if any(word in q for word in incident_words):
        return _answer_incidents(question)

    # --------------------------------------------------------
    # PRECURSORS
    # --------------------------------------------------------

    if (
        "precursor" in q
        or "precursors" in q
    ):
        n = _get_number(question)
        metric = _get_metric(question)

        records = top_precursors(
            n=n,
            metric=metric
        )

        return _format_ranking(
            records,
            "precursor",
            metric
        )

    # --------------------------------------------------------
    # ACTIVITIES
    # --------------------------------------------------------

    if (
        "activity" in q
        or "activities" in q
    ):
        n = _get_number(question)
        metric = _get_metric(question)

        records = top_activities(
            n=n,
            metric=metric
        )

        return _format_ranking(
            records,
            "activity_group",
            metric
        )

    # --------------------------------------------------------
    # BARRIERS
    # --------------------------------------------------------

    if (
        "barrier" in q
        or "barriers" in q
    ):
        n = _get_number(question)
        metric = _get_metric(question)

        records = top_barriers(
            n=n,
            metric=metric
        )

        return _format_ranking(
            records,
            "barrier_theme",
            metric
        )

    # --------------------------------------------------------
    # LIFE-SAVING RULES
    # --------------------------------------------------------

    if (
        "life saving rule" in q
        or "life-saving rule" in q
        or "life saving rules" in q
        or "lsr" in q
    ):
        n = _get_number(question)
        metric = _get_metric(question)

        records = top_life_saving_rules(
            n=n,
            metric=metric
        )

        return _format_ranking(
            records,
            "life_saving_rule",
            metric
        )

    # --------------------------------------------------------
    # HELP
    # --------------------------------------------------------

    return """
### I can help you analyze the OIL SIF dataset

Try asking:

- **How many reports are in the dataset?**
- **What are the top 5 activities by SIF density?**
- **What are the top 10 precursors by SIF reports?**
- **Which barriers have the highest SIF density?**
- **What are the top 5 Life-Saving Rules?**
- **Find incidents involving forklifts**
- **Find incidents involving lifting**
- **Find incidents related to confined space**
- **What is a Safety Instrumented Function?**
""".strip()