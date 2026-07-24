# Real-Data Orbital Congestion Monitoring

This repository contains Coco/Ririko's single master notebook and Streamlit dashboard for a real-data orbital congestion research project.

## Scientific scope

The project identifies orbital objects that occupy similar regimes using catalogue-level features. It estimates relative feature-space crowding through clustering and nearest-neighbour analysis. It does **not** calculate collision probability, propagate trajectories, or perform conjunction analysis.

## Real data source

The notebook uses the official CelesTrak SATCAT CSV:

- Catalogue: https://celestrak.org/satcat/
- Format documentation: https://celestrak.org/satcat/satcat-format.php
- Raw CSV: https://celestrak.org/pub/satcat.csv

A real snapshot is committed at `data/satcat.csv`. The notebook attempts a live refresh and falls back to the committed real snapshot if the request fails. It never substitutes synthetic or embedded demonstration data.

## Repository structure

```text
coco-orbital-congestion-real-data/
├── README.md
├── app.py
├── requirements.txt
├── runtime.txt
├── .gitignore
├── .streamlit/
│   └── config.toml
├── data/
│   ├── README.md
│   └── satcat.csv
├── notebooks/
│   └── coco_master_orbital_congestion_real_data.ipynb
└── outputs/
    ├── figures/
    └── tables/
```

## Models and analyses included

The master notebook contains:

- data acquisition and validation,
- cleaning and LEO filtering,
- feature engineering,
- StandardScaler feature scaling,
- pilot-versus-full real-catalogue comparison,
- K-Means clustering,
- silhouette-score and inertia evaluation,
- PCA visualization,
- nearest-neighbour analysis,
- a 0-to-100 congestion percentile score,
- debris-to-payload feature-space screening,
- owner-level descriptive summaries,
- exploratory Random Forest classification,
- paper-ready figures and tables, and
- Streamlit-compatible exports.

## Important paper revision

The earlier 28-record demonstration metrics are superseded. The final paper must use the values printed in Section 18 of the master notebook after a clean run. Do not retain the earlier 88% accuracy, 0.86 silhouette score, 96.77% PCA variance, or 28-record sample size unless a separate historical comparison explicitly labels them as demonstration results.

## Mac local setup

Open Terminal and move into the repository folder:

```bash
cd path/to/coco-orbital-congestion-real-data
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the master notebook:

```bash
jupyter notebook notebooks/coco_master_orbital_congestion_real_data.ipynb
```

Run every cell from top to bottom. The notebook writes all app-ready files into `outputs/tables/` and figures into `outputs/figures/`.

Run the dashboard:

```bash
python -m streamlit run app.py
```

## Streamlit Cloud deployment

1. Push the complete repository to GitHub.
2. Confirm `app.py`, `requirements.txt`, `data/satcat.csv`, and `outputs/tables/dashboard_ready_table.csv` are visible on the deployed branch.
3. Create a Streamlit Community Cloud app using `app.py` as the entrypoint.
4. Reboot the app after any data or code commit.

The dashboard reads notebook exports. It does not hard-code research results.

## Main dashboard outputs

- `dashboard_ready_table.csv`
- `cluster_summary.csv`
- `nearest_neighbor_table.csv`
- `debris_payload_pairs.csv`
- `owner_summary.csv`
- `random_forest_metrics.csv`
- `random_forest_feature_importance.csv`
- `model_summary.json`
- `data_provenance.json`

## Interpretation rules

- A high congestion score means high relative crowding in the selected standardized feature space.
- A nearest debris-payload pair is not a confirmed physical conjunction.
- Silhouette score measures cluster cohesion and separation, not classification accuracy.
- PCA explained variance is not model accuracy.
- Owner-level summaries do not assign responsibility or liability.
