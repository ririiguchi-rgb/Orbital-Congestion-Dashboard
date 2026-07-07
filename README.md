# Orbital Congestion Monitoring Dashboard

This repository contains Coco/Ririko's master notebook and Streamlit dashboard for an orbital congestion monitoring research project.

## Scientific scope

This project does **not** predict real satellite collision probability. It does not perform conjunction analysis, orbital propagation, covariance modeling, or time-dependent position/velocity prediction.

Instead, the project estimates simplified orbital congestion using catalog-level orbital feature similarity. Objects are compared using period, inclination, mean altitude, and orbit spread. The resulting congestion score is a monitoring proxy for identifying objects in crowded orbital feature-space regions.

## Repository structure

```text
coco-orbital-congestion-dashboard/
├── README.md
├── app.py
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── config.toml
├── data/
│   └── coco_demo_current_columns.csv
├── notebooks/
│   └── coco_master_orbital_congestion.ipynb
└── outputs/
    ├── figures/
    └── tables/
```

The dashboard can run from the included demo CSV. The notebook also exports processed tables and figures into `outputs/`.

## Expected CSV columns

The raw CSV should contain these columns:

```text
OBJECT_NAME, OBJECT_ID, NORAD_CAT_ID, OBJECT_TYPE, OPS_STATUS_CODE, OWNER, LAUNCH_DATE, LAUNCH_SITE, DECAY_DATE, PERIOD, INCLINATION, APOGEE, PERIGEE, RCS, DATA_STATUS_CODE, ORBIT_CENTER, ORBIT_TYPE
```

To analyze a larger real catalog, save it as:

```text
data/satcat.csv
```

If `data/satcat.csv` is missing, the notebook uses:

```text
data/coco_demo_current_columns.csv
```

## Mac setup

Open Terminal and move into the repository folder:

```bash
cd path/to/coco-orbital-congestion-dashboard
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the notebook

```bash
jupyter notebook notebooks/coco_master_orbital_congestion.ipynb
```

Run all cells. The notebook will create or update:

```text
outputs/tables/
outputs/figures/
```

## Run the Streamlit dashboard locally

```bash
python -m streamlit run app.py
```

The app works from the included demo CSV and also allows users to upload a SATCAT-style CSV from the sidebar.

## Correct interpretation of the nearest-neighbor matrix

If the notebook or dashboard reports a nearest-neighbor matrix shape such as `(28, 4)`, this means:

- 28 complete orbital records were used after cleaning, and
- 4 nearest neighbors were requested for each object.

This is not an error and does not mean the model has only four data points.

## Paper wording

Use careful language in the paper:

- Correct: "This study estimates simplified orbital congestion using feature-space similarity."
- Correct: "Objects with similar orbital features may be useful monitoring candidates."
- Incorrect: "This model predicts real collision probability."
