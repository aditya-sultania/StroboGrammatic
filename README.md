# OIL SIF Precursor Intelligence — Prototype v5

This version keeps the v4 workflow and improves the Analyze/XAI experience.

## User-facing analysis
- SIF probability
- SIF / Non-SIF decision
- High-risk safety backstop
- Simple human-readable explanation
- Detected activity, precursor, barrier and LSR
- Recommended HSE action
- Technical XAI in an expandable section
- Safety-oriented review level

## Dashboard workflow
Overview -> SIF Risk -> Site / Activity Ranking -> Life-Saving Rules -> Precursors -> Barrier Failures -> XAI / Analyze -> HSE Review Queue

## Run
pip install -r requirements.txt
streamlit run app.py

## Important
Public OSHA data is used for the prototype. OIL site hierarchy requires OIL HSSE data.
The simple explanation is an interpretation of model/rule outputs; it is not a causal explanation of an incident.
Development model probabilities/metrics are not expert-validated production safety probabilities.
