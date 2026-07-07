"""
Streamlit dashboard for Coco/Ririko's orbital congestion monitoring project.

Scientific scope:
This app performs simplified catalog-level congestion screening using orbital feature
similarity. It does not calculate real collision probability, conjunction events,
or time-dependent orbital propagation.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Orbital Congestion Dashboard",
    page_icon="🛰️",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "synthetic_data.csv"
OUTPUT_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_COLUMNS = [
    "OBJECT_NAME", "OBJECT_ID", "NORAD_CAT_ID", "OBJECT_TYPE", "OPS_STATUS_CODE",
    "OWNER", "LAUNCH_DATE", "LAUNCH_SITE", "DECAY_DATE", "PERIOD", "INCLINATION",
    "APOGEE", "PERIGEE", "RCS", "DATA_STATUS_CODE", "ORBIT_CENTER", "ORBIT_TYPE",
]

CORE_FEATURES = ["PERIOD", "INCLINATION", "MEAN_ALTITUDE", "ORBIT_SPREAD"]
REQUESTED_NEIGHBORS = 4
RANDOM_STATE = 42


def normalize_object_type(value: object) -> str:
    """Map object-type labels into broad, paper-friendly categories."""
    text = str(value).strip().upper()
    if text in {"PAY", "PAYLOAD", "SAT", "SATELLITE"}:
        return "PAY"
    if text in {"DEB", "DEBRIS"}:
        return "DEB"
    if text in {"R/B", "RB", "ROCKET BODY", "ROCKET_BODY"}:
        return "R/B"
    if text in {"UNK", "UNKNOWN", "NAN", ""}:
        return "UNKNOWN"
    return text


@st.cache_data
def load_default_csv() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_DATA_PATH)


@st.cache_data
def load_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(StringIO(file_bytes.decode("utf-8")))


def validate_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    extra = [col for col in df.columns if col not in EXPECTED_COLUMNS]
    return df, missing, extra


def prepare_dataset(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean the catalog, run the unsupervised workflow, and return dashboard tables."""
    df = df_raw.copy()
    df.columns = [str(col).strip() for col in df.columns]

    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    numeric_columns = ["NORAD_CAT_ID", "PERIOD", "INCLINATION", "APOGEE", "PERIGEE", "RCS"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["LAUNCH_DATE", "DECAY_DATE"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["OBJECT_TYPE"] = df["OBJECT_TYPE"].apply(normalize_object_type)
    df["OWNER"] = df["OWNER"].fillna("UNKNOWN").astype(str).str.strip().replace("", "UNKNOWN")
    df["OBJECT_NAME"] = df["OBJECT_NAME"].fillna("Unnamed object").astype(str).str.strip()

    df["MEAN_ALTITUDE"] = (df["APOGEE"] + df["PERIGEE"]) / 2
    df["ORBIT_SPREAD"] = df["APOGEE"] - df["PERIGEE"]
    df["ALTITUDE_RATIO"] = df["APOGEE"] / df["PERIGEE"].replace(0, np.nan)
    df["HAS_DECAY_DATE"] = df["DECAY_DATE"].notna().astype(int)
    df["IS_OPERATIONAL"] = df["OPS_STATUS_CODE"].fillna("").astype(str).str.strip().eq("+").astype(int)
    df["IS_DEBRIS"] = df["OBJECT_TYPE"].eq("DEB").astype(int)
    df["IS_PAYLOAD"] = df["OBJECT_TYPE"].eq("PAY").astype(int)

    required_for_analysis = ["PERIOD", "INCLINATION", "APOGEE", "PERIGEE", "MEAN_ALTITUDE", "ORBIT_SPREAD"]
    analysis_df = df.dropna(subset=required_for_analysis).copy()
    analysis_df = analysis_df[(analysis_df["PERIOD"] > 0) & (analysis_df["APOGEE"] >= analysis_df["PERIGEE"])]
    analysis_df = analysis_df.reset_index(drop=True)

    if len(analysis_df) < 2:
        raise ValueError("At least two complete orbital records are required for nearest-neighbor analysis.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(analysis_df[CORE_FEATURES])

    max_k = min(8, len(analysis_df) - 1)
    cluster_rows: list[dict[str, float | int | str]] = []
    best_k = 1
    best_score = np.nan

    if max_k >= 2:
        for k in range(2, max_k + 1):
            model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            labels = model.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else np.nan
            cluster_rows.append({"k": k, "inertia": float(model.inertia_), "silhouette_score": float(score)})
        cluster_eval = pd.DataFrame(cluster_rows)
        best_row = cluster_eval.sort_values("silhouette_score", ascending=False).iloc[0]
        best_k = int(best_row["k"])
        best_score = float(best_row["silhouette_score"])
        final_model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
        analysis_df["CLUSTER"] = final_model.fit_predict(X_scaled).astype(str)
    else:
        cluster_eval = pd.DataFrame([{"k": 1, "inertia": 0.0, "silhouette_score": np.nan}])
        analysis_df["CLUSTER"] = "0"

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    analysis_df["PC1"] = X_pca[:, 0]
    analysis_df["PC2"] = X_pca[:, 1]
    analysis_df["PCA_VARIANCE_SUM"] = float(pca.explained_variance_ratio_.sum())

    neighbor_count = min(REQUESTED_NEIGHBORS, len(analysis_df) - 1)
    nearest_model = NearestNeighbors(n_neighbors=neighbor_count + 1, metric="euclidean")
    nearest_model.fit(X_scaled)
    distances_with_self, indices_with_self = nearest_model.kneighbors(X_scaled)
    neighbor_distances = distances_with_self[:, 1:]
    neighbor_indices = indices_with_self[:, 1:]

    analysis_df["AVG_NN_DISTANCE"] = neighbor_distances.mean(axis=1)
    analysis_df["CONGESTION_SCORE"] = 1 / (analysis_df["AVG_NN_DISTANCE"] + 1e-6)
    analysis_df["CONGESTION_PERCENTILE"] = analysis_df["CONGESTION_SCORE"].rank(pct=True) * 100

    high_cutoff = analysis_df["CONGESTION_SCORE"].quantile(0.75)
    medium_cutoff = analysis_df["CONGESTION_SCORE"].quantile(0.40)
    analysis_df["CONGESTION_CATEGORY"] = np.select(
        [
            analysis_df["CONGESTION_SCORE"] >= high_cutoff,
            analysis_df["CONGESTION_SCORE"] >= medium_cutoff,
        ],
        ["High", "Medium"],
        default="Low",
    )

    neighbor_records = []
    for row_i, object_row in analysis_df.iterrows():
        for rank, (neighbor_i, distance) in enumerate(zip(neighbor_indices[row_i], neighbor_distances[row_i]), start=1):
            neighbor_row = analysis_df.iloc[int(neighbor_i)]
            neighbor_records.append(
                {
                    "OBJECT_NAME": object_row["OBJECT_NAME"],
                    "OBJECT_TYPE": object_row["OBJECT_TYPE"],
                    "OWNER": object_row["OWNER"],
                    "NEIGHBOR_RANK": rank,
                    "NEIGHBOR_NAME": neighbor_row["OBJECT_NAME"],
                    "NEIGHBOR_TYPE": neighbor_row["OBJECT_TYPE"],
                    "NEIGHBOR_OWNER": neighbor_row["OWNER"],
                    "FEATURE_SPACE_DISTANCE": float(distance),
                }
            )
    neighbor_table = pd.DataFrame(neighbor_records)

    payload_df = analysis_df[analysis_df["OBJECT_TYPE"].eq("PAY")].copy()
    debris_df = analysis_df[analysis_df["OBJECT_TYPE"].eq("DEB")].copy()
    pair_records = []
    if len(payload_df) > 0 and len(debris_df) > 0:
        payload_positions = analysis_df.index[analysis_df["OBJECT_TYPE"].eq("PAY")].to_numpy()
        debris_positions = analysis_df.index[analysis_df["OBJECT_TYPE"].eq("DEB")].to_numpy()
        payload_scaled = X_scaled[payload_positions]
        debris_scaled = X_scaled[debris_positions]
        payload_nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        payload_nn.fit(payload_scaled)
        pair_distances, pair_indices = payload_nn.kneighbors(debris_scaled)
        for debris_pos, local_payload_index, distance in zip(debris_positions, pair_indices[:, 0], pair_distances[:, 0]):
            payload_pos = payload_positions[int(local_payload_index)]
            debris_row = analysis_df.iloc[int(debris_pos)]
            payload_row = analysis_df.iloc[int(payload_pos)]
            pair_records.append(
                {
                    "DEBRIS_OBJECT": debris_row["OBJECT_NAME"],
                    "DEBRIS_OWNER": debris_row["OWNER"],
                    "NEAREST_PAYLOAD": payload_row["OBJECT_NAME"],
                    "PAYLOAD_OWNER": payload_row["OWNER"],
                    "FEATURE_SPACE_DISTANCE": float(distance),
                    "DEBRIS_CLUSTER": debris_row["CLUSTER"],
                    "PAYLOAD_CLUSTER": payload_row["CLUSTER"],
                }
            )
    debris_payload_pairs = pd.DataFrame(pair_records).sort_values("FEATURE_SPACE_DISTANCE") if pair_records else pd.DataFrame()

    cluster_summary = (
        analysis_df.groupby("CLUSTER")
        .agg(
            OBJECT_COUNT=("OBJECT_NAME", "count"),
            MEAN_PERIOD=("PERIOD", "mean"),
            MEAN_INCLINATION=("INCLINATION", "mean"),
            MEAN_ALTITUDE=("MEAN_ALTITUDE", "mean"),
            MEAN_ORBIT_SPREAD=("ORBIT_SPREAD", "mean"),
            MEAN_CONGESTION_SCORE=("CONGESTION_SCORE", "mean"),
            DEBRIS_COUNT=("IS_DEBRIS", "sum"),
            PAYLOAD_COUNT=("IS_PAYLOAD", "sum"),
        )
        .reset_index()
        .sort_values("CLUSTER")
    )

    analysis_df.attrs["nearest_neighbor_shape"] = neighbor_distances.shape
    analysis_df.attrs["best_k"] = best_k
    analysis_df.attrs["best_silhouette_score"] = best_score
    analysis_df.attrs["rows_used"] = len(analysis_df)
    analysis_df.attrs["rows_raw"] = len(df_raw)

    return analysis_df, neighbor_table, debris_payload_pairs, cluster_eval, cluster_summary


def save_tables(*tables: tuple[str, pd.DataFrame]) -> None:
    for name, table in tables:
        table.to_csv(OUTPUT_TABLE_DIR / name, index=False)


st.title("Orbital Congestion Monitoring Dashboard")
st.caption(
    "A Streamlit dashboard for unsupervised orbital feature clustering, nearest-neighbor analysis, "
    "and simplified congestion screening. This is not a real collision-probability calculator."
)

with st.expander("Scientific scope and limitation", expanded=True):
    st.write(
        "This dashboard identifies catalog objects that occupy similar orbital regimes based on static features "
        "such as period, inclination, apogee, perigee, mean altitude, and orbit spread. The congestion score is a "
        "feature-space monitoring proxy. Real collision-risk analysis would require time-dependent state vectors, "
        "orbital propagation, covariance/uncertainty information, conjunction screening, and object-size modeling."
    )

uploaded_file = st.sidebar.file_uploader("Optional: upload a SATCAT-style CSV", type=["csv"])
if uploaded_file is not None:
    raw_df = load_uploaded_csv(uploaded_file.getvalue())
    data_label = uploaded_file.name
else:
    raw_df = load_default_csv()
    data_label = "data/coco_demo_current_columns.csv"

raw_df, missing_columns, extra_columns = validate_columns(raw_df)

st.sidebar.header("Dataset status")
st.sidebar.write(f"Source: `{data_label}`")
st.sidebar.write(f"Raw rows: **{len(raw_df)}**")
if missing_columns:
    st.sidebar.error("Missing required columns: " + ", ".join(missing_columns))
    st.stop()
if extra_columns:
    st.sidebar.info("Extra columns detected and ignored: " + ", ".join(extra_columns))

try:
    dashboard_df, neighbor_df, debris_payload_df, cluster_eval_df, cluster_summary_df = prepare_dataset(raw_df)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

save_tables(
    ("dashboard_ready_orbital_congestion.csv", dashboard_df),
    ("nearest_neighbor_table.csv", neighbor_df),
    ("debris_payload_pairs.csv", debris_payload_df),
    ("cluster_evaluation.csv", cluster_eval_df),
    ("cluster_summary.csv", cluster_summary_df),
)

shape = dashboard_df.attrs.get("nearest_neighbor_shape", (len(dashboard_df), min(REQUESTED_NEIGHBORS, max(len(dashboard_df) - 1, 0))))
with st.expander("Correct interpretation of the nearest-neighbor matrix", expanded=False):
    st.write(
        f"The nearest-neighbor distance matrix shape for the current dataset is **{shape}**. "
        f"This means **{shape[0]} analyzed objects** and **{shape[1]} nearest neighbors per object**. "
        "It does not mean there are only four data points."
    )

# Sidebar filters after processing
st.sidebar.header("Dashboard filters")
object_types = sorted(dashboard_df["OBJECT_TYPE"].dropna().unique().tolist())
owners = sorted(dashboard_df["OWNER"].dropna().unique().tolist())
clusters = sorted(dashboard_df["CLUSTER"].dropna().unique().tolist())
categories = ["Low", "Medium", "High"]

selected_types = st.sidebar.multiselect("Object type", object_types, default=object_types)
selected_owners = st.sidebar.multiselect("Owner", owners, default=owners)
selected_clusters = st.sidebar.multiselect("Cluster", clusters, default=clusters)
selected_categories = st.sidebar.multiselect("Congestion category", categories, default=categories)

filtered = dashboard_df[
    dashboard_df["OBJECT_TYPE"].isin(selected_types)
    & dashboard_df["OWNER"].isin(selected_owners)
    & dashboard_df["CLUSTER"].isin(selected_clusters)
    & dashboard_df["CONGESTION_CATEGORY"].isin(selected_categories)
].copy()

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Raw records", len(raw_df))
metric_2.metric("Analyzed records", len(dashboard_df))
metric_3.metric("Objects shown", len(filtered))
metric_4.metric("Best k", dashboard_df.attrs.get("best_k", "N/A"))

metric_5, metric_6, metric_7, metric_8 = st.columns(4)
metric_5.metric("High congestion objects", int((filtered["CONGESTION_CATEGORY"] == "High").sum()) if not filtered.empty else 0)
metric_6.metric("Max congestion score", f"{filtered['CONGESTION_SCORE'].max():.2f}" if not filtered.empty else "N/A")
metric_7.metric("PCA variance shown", f"{dashboard_df['PCA_VARIANCE_SUM'].iloc[0] * 100:.2f}%")
metric_8.metric("Neighbor matrix", f"{shape[0]} × {shape[1]}")

tab_overview, tab_clusters, tab_congestion, tab_pairs, tab_data = st.tabs(
    ["Overview", "Clusters", "Congestion", "Debris-Payload Pairs", "Data"]
)

with tab_overview:
    st.subheader("Orbital regime overview")
    if filtered.empty:
        st.warning("No objects match the selected filters.")
    else:
        fig = px.scatter(
            filtered,
            x="INCLINATION",
            y="MEAN_ALTITUDE",
            color="CONGESTION_CATEGORY",
            symbol="OBJECT_TYPE",
            hover_data=["OBJECT_NAME", "OWNER", "CLUSTER", "PERIOD", "APOGEE", "PERIGEE", "CONGESTION_SCORE"],
            labels={"INCLINATION": "Inclination [deg]", "MEAN_ALTITUDE": "Mean altitude [km]"},
            title="Objects by Inclination, Mean Altitude, and Congestion Category",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.write("Object type counts")
        type_counts = filtered["OBJECT_TYPE"].value_counts().reset_index()
        type_counts.columns = ["OBJECT_TYPE", "COUNT"]
        st.dataframe(type_counts, use_container_width=True)

with tab_clusters:
    st.subheader("PCA cluster map")
    if filtered.empty:
        st.warning("No objects match the selected filters.")
    else:
        fig = px.scatter(
            filtered,
            x="PC1",
            y="PC2",
            color="CLUSTER",
            symbol="OBJECT_TYPE",
            hover_data=["OBJECT_NAME", "OWNER", "CONGESTION_SCORE", "CONGESTION_CATEGORY"],
            title="PCA View of K-Means Clusters",
        )
        st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.write("Cluster evaluation")
        st.dataframe(cluster_eval_df, use_container_width=True)
    with col_b:
        st.write("Cluster summary")
        st.dataframe(cluster_summary_df, use_container_width=True)

with tab_congestion:
    st.subheader("Top objects by simplified congestion score")
    if filtered.empty:
        st.warning("No objects match the selected filters.")
    else:
        max_slider = max(5, min(30, len(filtered)))
        top_n = st.slider("Number of objects to show", min_value=5, max_value=max_slider, value=min(10, max_slider))
        top_df = filtered.sort_values("CONGESTION_SCORE", ascending=False).head(top_n)
        fig = px.bar(
            top_df.sort_values("CONGESTION_SCORE"),
            x="CONGESTION_SCORE",
            y="OBJECT_NAME",
            orientation="h",
            color="CONGESTION_CATEGORY",
            hover_data=["OBJECT_TYPE", "OWNER", "CLUSTER", "AVG_NN_DISTANCE"],
            title="Highest Feature-Space Congestion Scores",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            top_df[["OBJECT_NAME", "OBJECT_TYPE", "OWNER", "CLUSTER", "AVG_NN_DISTANCE", "CONGESTION_SCORE", "CONGESTION_PERCENTILE", "CONGESTION_CATEGORY"]],
            use_container_width=True,
        )

with tab_pairs:
    st.subheader("Debris-payload proximity pairs")
    st.write(
        "These pairs show debris objects and their nearest payload objects in standardized orbital feature space. "
        "They are monitoring candidates, not confirmed close approaches."
    )
    if debris_payload_df.empty:
        st.info("No debris-payload proximity pairs were identified in the current analyzed dataset.")
    else:
        st.dataframe(debris_payload_df, use_container_width=True)

with tab_data:
    st.subheader("Filtered dashboard data")
    st.dataframe(filtered, use_container_width=True)
    st.download_button(
        "Download filtered dashboard data",
        data=filtered.to_csv(index=False),
        file_name="filtered_orbital_congestion_dashboard.csv",
        mime="text/csv",
    )

    st.subheader("Nearest-neighbor table")
    st.dataframe(neighbor_df, use_container_width=True)
    st.download_button(
        "Download nearest-neighbor table",
        data=neighbor_df.to_csv(index=False),
        file_name="nearest_neighbor_table.csv",
        mime="text/csv",
    )

