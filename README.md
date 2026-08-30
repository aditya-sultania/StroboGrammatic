# OIL SIF Sentinel — SIH 2026 Finalist Prototype Pack

This is a working Streamlit prototype built around the three supplied datasets.

## What is included

- Three role-restricted dashboards: Worker, HSE Safety Officer, Management.
- HSE command center using the SIF master dataset.
- State-level management ranking using the OSHA incident dataset's actual `State` field.
- State hazard drill-downs using `NatureTitle` / `EventTitle`.
- Transparent state risk indicator based on incident burden + severity burden.
- SIF queue with probability, precursor, Life-Saving Rule, review status and priority score.
- Trend and precursor analytics.
- Dataset-grounded HSE AI assistant.
- Demo authentication for the prototype.

## Run

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## Demo accounts

- Worker: `worker` / `worker123`
- HSE: `hse` / `hse123`
- Management: `manager` / `manager123`

These are NOT production credentials.

## Dataset placement

All three CSVs are expected in:

`data/raw/`

Required filenames:

- `January2015toNovember2025.csv`
- `SIF_Dashboard_Full_ITA_Inference_v2.csv`
- `SIF_Dashboard_Master_v4.csv`

The original uploaded filenames had `(3)`, `(1)`, etc. Those suffixes are removed in this project so the code has deterministic paths.

## Important analytical boundary

The OSHA incident dataset contains a real `State` column and is therefore used for state-level ranking and hazard analysis.

The SIF Master dataset does not contain a state field in the supplied schema. Therefore the app does NOT fabricate a state-to-SIF mapping. SIF inference is analyzed globally in the HSE workspace, while state hazard/incident intelligence comes from the OSHA dataset.

The state risk index is a prototype decision-support score, not a regulatory or actuarial risk rating.

## Integrating into the existing StroboGrammatic repository

Copy these files/folders into the existing repository:

- `app.py` → replace only if you want this app as the main entry point.
- `src/data_loader.py`
- `src/analytics.py`
- `src/assistant.py`
- `data/raw/*.csv`
- `requirements.txt` → merge dependencies if your existing project has additional packages.

If your current project already has valuable pages/modules, do not delete them. Instead import these modules into the existing app and route the three roles to them.
