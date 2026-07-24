from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
DEFAULT_DASHBOARD_PATH = TABLE_DIR / "dashboard_ready_table.csv"
MODEL_SUMMARY_PATH = TABLE_DIR / "model_summary.json"
CLUSTER_SUMMARY_PATH = TABLE_DIR / "cluster_summary.csv"
PAIR_PATH = TABLE_DIR / "debris_payload_pairs.csv"
RF_METRICS_PATH = TABLE_DIR / "random_forest_metrics.csv"
RF_IMPORTANCE_PATH = TABLE_DIR / "random_forest_feature_importance.csv"
PROVENANCE_PATH = TABLE_DIR / "data_provenance.json"

REQUIRED_COLUMNS = {
    "OBJECT_NAME",
    "NORAD_CAT_ID",
    "OBJECT_TYPE",
    "OWNER",
    "CLUSTER",
    "PCA1",
    "PCA2",
    "MEAN_ALTITUDE",
    "INCLINATION",
    "CONGESTION_SCORE",
    "CONGESTION_CATEGORY",
}

st.set_page_config(
    page_title="Orbital Congestion Monitoring",
    page_icon="🛰️",
    layout="wide",
)


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return load_csv(path)


st.title("Orbital Congestion Monitoring Dashboard")
st.caption("Real CelesTrak SATCAT data with clustering, nearest-neighbour screening, and exploratory classification")
st.warning(
    "Scientific scope: this dashboard shows relative orbital feature-space crowding. "
    "It does not calculate collision probability or predict physical conjunctions."
)

with st.sidebar:
    st.header("Data")
    uploaded_file = st.file_uploader(
        "Optional dashboard-ready CSV",
        type=["csv"],
        help="Upload a CSV exported by the master notebook. Raw SATCAT files must first be processed by the notebook.",
    )

if uploaded_file is not None:
    dashboard_df = pd.read_csv(uploaded_file)
    source_label = "Uploaded dashboard-ready table"
elif DEFAULT_DASHBOARD_PATH.exists():
    dashboard_df = load_csv(DEFAULT_DASHBOARD_PATH)
    source_label = str(DEFAULT_DASHBOARD_PATH.relative_to(PROJECT_ROOT))
else:
    st.error(
        "The dashboard-ready table is missing. Run the master notebook from top to bottom, "
        "commit outputs/tables/dashboard_ready_table.csv, and reboot the Streamlit app."
    )
    st.stop()

missing_columns = REQUIRED_COLUMNS.difference(dashboard_df.columns)
if missing_columns:
    st.error(f"The dashboard table is missing required columns: {sorted(missing_columns)}")
    st.stop()

for numeric_column in [
    "NORAD_CAT_ID",
    "CLUSTER",
    "PCA1",
    "PCA2",
    "MEAN_ALTITUDE",
    "INCLINATION",
    "CONGESTION_SCORE",
]:
    dashboard_df[numeric_column] = pd.to_numeric(dashboard_df[numeric_column], errors="coerce")

dashboard_df["OBJECT_TYPE"] = dashboard_df["OBJECT_TYPE"].fillna("UNKNOWN").astype(str)
dashboard_df["OWNER"] = dashboard_df["OWNER"].fillna("UNKNOWN").astype(str)
dashboard_df["CONGESTION_CATEGORY"] = dashboard_df["CONGESTION_CATEGORY"].fillna("Unknown").astype(str)

with st.sidebar:
    st.caption(f"Loaded: {source_label}")
    st.header("Filters")

    object_type_options = sorted(dashboard_df["OBJECT_TYPE"].dropna().unique().tolist())
    selected_types = st.multiselect(
        "Object type",
        object_type_options,
        default=object_type_options,
    )

    owner_options = sorted(dashboard_df["OWNER"].dropna().unique().tolist())
    selected_owners = st.multiselect(
        "Owner",
        owner_options,
        default=owner_options,
    )

    cluster_options = sorted(dashboard_df["CLUSTER"].dropna().astype(int).unique().tolist())
    selected_clusters = st.multiselect(
        "Cluster",
        cluster_options,
        default=cluster_options,
    )

    category_order = ["Very High", "High", "Moderate", "Low"]
    category_options = [
        category for category in category_order if category in dashboard_df["CONGESTION_CATEGORY"].unique()
    ]
    selected_categories = st.multiselect(
        "Congestion category",
        category_options,
        default=category_options,
    )

filtered_df = dashboard_df[
    dashboard_df["OBJECT_TYPE"].isin(selected_types)
    & dashboard_df["OWNER"].isin(selected_owners)
    & dashboard_df["CLUSTER"].isin(selected_clusters)
    & dashboard_df["CONGESTION_CATEGORY"].isin(selected_categories)
].copy()
filtered_df["CLUSTER_LABEL"] = filtered_df["CLUSTER"].round().astype("Int64").astype(str)

if filtered_df.empty:
    st.info("No records match the current filter selection.")
    st.stop()

model_summary = load_json(MODEL_SUMMARY_PATH) if MODEL_SUMMARY_PATH.exists() else {}
cluster_summary = optional_csv(CLUSTER_SUMMARY_PATH)
pairs_df = optional_csv(PAIR_PATH)
rf_metrics = optional_csv(RF_METRICS_PATH)
rf_importance = optional_csv(RF_IMPORTANCE_PATH)
provenance = load_json(PROVENANCE_PATH) if PROVENANCE_PATH.exists() else {}

metric_columns = st.columns(4)
metric_columns[0].metric("Filtered objects", f"{len(filtered_df):,}")
metric_columns[1].metric("Clusters shown", filtered_df["CLUSTER"].nunique())
metric_columns[2].metric("Median congestion score", f"{filtered_df['CONGESTION_SCORE'].median():.1f}")
metric_columns[3].metric(
    "High or very high",
    f"{filtered_df['CONGESTION_CATEGORY'].isin(['High', 'Very High']).sum():,}",
)

st.download_button(
    "Download filtered dashboard table",
    data=filtered_df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_orbital_congestion_table.csv",
    mime="text/csv",
)

overview_tab, cluster_tab, congestion_tab, pairs_tab, model_tab, methods_tab = st.tabs(
    [
        "Overview",
        "Cluster map",
        "Congestion rankings",
        "Debris-payload screening",
        "Model metrics",
        "Data and limitations",
    ]
)

with overview_tab:
    st.subheader("Orbital regimes")
    overview_plot = px.scatter(
        filtered_df,
        x="INCLINATION",
        y="MEAN_ALTITUDE",
        color="OBJECT_TYPE",
        hover_name="OBJECT_NAME",
        hover_data=["NORAD_CAT_ID", "OWNER", "CLUSTER", "CONGESTION_SCORE"],
        labels={
            "INCLINATION": "Inclination (degrees)",
            "MEAN_ALTITUDE": "Mean altitude (km)",
        },
        title="Mean altitude versus inclination",
        opacity=0.65,
    )
    st.plotly_chart(overview_plot, use_container_width=True)

    type_counts = filtered_df["OBJECT_TYPE"].value_counts().reset_index()
    type_counts.columns = ["OBJECT_TYPE", "COUNT"]
    type_plot = px.bar(
        type_counts,
        x="OBJECT_TYPE",
        y="COUNT",
        title="Filtered object-type composition",
    )
    st.plotly_chart(type_plot, use_container_width=True)

with cluster_tab:
    st.subheader("PCA visualization of K-Means clusters")
    pca_plot = px.scatter(
        filtered_df,
        x="PCA1",
        y="PCA2",
        color="CLUSTER_LABEL",
        symbol="OBJECT_TYPE",
        hover_name="OBJECT_NAME",
        hover_data=["NORAD_CAT_ID", "OWNER", "MEAN_ALTITUDE", "CONGESTION_SCORE"],
        labels={"CLUSTER_LABEL": "Cluster"},
        title="Two-dimensional PCA projection",
        opacity=0.65,
    )
    st.plotly_chart(pca_plot, use_container_width=True)
    st.caption(
        "PCA is a visualization of the selected standardized features. "
        "Its explained variance is not model accuracy."
    )

    if not cluster_summary.empty:
        st.subheader("Cluster summary")
        st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

with congestion_tab:
    st.subheader("Highest relative feature-space congestion")
    top_n = st.slider("Number of objects to show", min_value=10, max_value=100, value=25, step=5)
    ranking = filtered_df.nlargest(top_n, "CONGESTION_SCORE").copy()
    ranking_plot = px.bar(
        ranking.sort_values("CONGESTION_SCORE"),
        x="CONGESTION_SCORE",
        y="OBJECT_NAME",
        orientation="h",
        color="CONGESTION_CATEGORY",
        hover_data=["OBJECT_TYPE", "OWNER", "CLUSTER", "MEAN_ALTITUDE"],
        title="Congestion percentile ranking",
        labels={"CONGESTION_SCORE": "Congestion score (0-100 percentile)"},
    )
    st.plotly_chart(ranking_plot, use_container_width=True)
    st.dataframe(
        ranking[
            [
                "OBJECT_NAME",
                "NORAD_CAT_ID",
                "OBJECT_TYPE",
                "OWNER",
                "CLUSTER",
                "MEAN_ALTITUDE",
                "INCLINATION",
                "CONGESTION_SCORE",
                "CONGESTION_CATEGORY",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

with pairs_tab:
    st.subheader("Nearest payload for each debris record")
    st.info(
        "These pairs are nearest in standardized orbital feature space. "
        "They are not confirmed physical conjunctions."
    )
    if pairs_df.empty:
        st.warning("The debris-payload screening table is not available. Run the master notebook again.")
    else:
        pair_limit = st.slider("Pairs to display", 10, 200, 50, 10)
        st.dataframe(pairs_df.head(pair_limit), use_container_width=True, hide_index=True)

with model_tab:
    st.subheader("Final model summary")
    if model_summary:
        summary_display = {
            "Real rows analysed": model_summary.get("analysis_rows"),
            "Selected K-Means k": model_summary.get("selected_k"),
            "Best silhouette score": model_summary.get("best_silhouette_score"),
            "PCA variance retained in 2D": model_summary.get("pca_explained_variance_total"),
            "Random Forest accuracy": model_summary.get("random_forest_accuracy"),
            "Random Forest balanced accuracy": model_summary.get("random_forest_balanced_accuracy"),
            "Random Forest macro F1": model_summary.get("random_forest_macro_f1"),
        }
        st.json(summary_display)

    if not rf_metrics.empty:
        st.subheader("Random Forest metrics")
        metrics_plot = px.bar(
            rf_metrics,
            x="METRIC",
            y="VALUE",
            title="Exploratory PAY-versus-DEB classification metrics",
        )
        metrics_plot.update_yaxes(range=[0, 1])
        st.plotly_chart(metrics_plot, use_container_width=True)
        st.dataframe(rf_metrics, use_container_width=True, hide_index=True)

    if not rf_importance.empty:
        importance_plot = px.bar(
            rf_importance.sort_values("IMPORTANCE"),
            x="IMPORTANCE",
            y="FEATURE",
            orientation="h",
            title="Random Forest feature importance",
        )
        st.plotly_chart(importance_plot, use_container_width=True)

with methods_tab:
    st.subheader("Data provenance")
    if provenance:
        st.json(provenance)
    else:
        st.write("Source: CelesTrak SATCAT CSV")

    st.subheader("Method summary")
    st.markdown(
        """
        1. Load a real CelesTrak SATCAT snapshot.
        2. Keep current Earth-centred LEO records with complete orbital fields.
        3. Engineer mean altitude, orbit spread, and altitude ratio.
        4. Standardize five orbital features.
        5. Select K-Means cluster count using silhouette score.
        6. Use PCA to display the feature space in two dimensions.
        7. Rank local crowding using nearest-neighbour distance percentiles.
        8. Screen debris records against their nearest payload in feature space.
        9. Run a separate exploratory Random Forest PAY-versus-DEB classification.
        """
    )

    st.subheader("Limitations")
    st.markdown(
        """
        - Similarity in catalogue features does not mean two objects occupy the same physical location at the same time.
        - The workflow does not propagate orbits, model covariance, or calculate conjunction probability.
        - The congestion score is relative to this filtered dataset and selected feature definition.
        - Owner summaries support filtering and transparency but do not assign responsibility or liability.
        - The Random Forest model is exploratory and separate from the main unsupervised workflow.
        """
    )
