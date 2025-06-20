import re

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from tennis_racquet_analysis.modeling.kmeans_utils import fit_kmeans
from tennis_racquet_analysis.preprocessing_utils import compute_pca_summary

# ─── 1) Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="K-Means Clustering Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.sidebar.title("Dashboard Settings")

# remember if we've run k-means
if "did_cluster" not in st.session_state:
    st.session_state.did_cluster = False

# ─── 2) Require upload ─────────────────────────────────────────────────────────
uploaded = st.sidebar.file_uploader(
    "Upload your own CSV", type="csv", help="Choose any local CSV to visualize"
)
if not uploaded:
    st.error("Please upload a CSV file to visualize.")
    st.stop()

df = pd.read_csv(uploaded)
dataset_label = uploaded.name

# 2a) lock in features for exact CLI-matching
original_feature_cols = df.select_dtypes(include="number").columns.tolist()

# ─── 2b) Detect any pre-existing cluster column ─────────────────────────────────
initial_clusters = [c for c in df.columns if re.search(r"cluster", c, re.I)]
if initial_clusters:
    st.session_state.did_cluster = False
    st.error(
        "Cannot run k-means: data file already contains cluster column(s): "
        f"{', '.join(initial_clusters)}"
    )

    # pick which existing cluster column to drive everything
    cluster_col = st.sidebar.selectbox("Cluster column", initial_clusters)
    # keep it numeric
    df[cluster_col] = df[cluster_col].astype(int)

    st.subheader("Imported Data (with existing clusters)")
    st.dataframe(df, use_container_width=True)

else:
    # ─── 3) Model Settings ───────────────────────────────────────────────────────
    st.sidebar.header("Model Settings")
    n_clusters = st.sidebar.slider("Number of clusters", 2, 15, 3, help="k for k-means")
    n_init = st.sidebar.number_input("n_init (k-means)", min_value=1, value=50, step=1)
    algo_method = st.sidebar.selectbox("Algorithm Method", ["lloyd", "elkan"])
    init_method = st.sidebar.selectbox("Init Method", ["k-means++", "random"])
    use_random_seed = st.sidebar.checkbox("Specify Random Seed", value=False)
    random_seed = None
    if use_random_seed:
        random_seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)
    run_cluster = st.sidebar.button("Run K-Means")

    # ─── 4) Run or pick clusters ────────────────────────────────────────────────
    if run_cluster:
        df = fit_kmeans(
            df,
            k=n_clusters,
            feature_columns=original_feature_cols,  # ← exact same features/order
            init=init_method,
            n_init=n_init,
            random_state=(random_seed if use_random_seed else None),
            algorithm=algo_method,
            label_column="cluster",
        )
        st.session_state.did_cluster = True

        # cluster_col stays 0-based numeric
        cluster_col = f"cluster_{n_clusters}"
        df[cluster_col] = df[cluster_col].astype(int)

        st.subheader("Clustered Data")
        st.dataframe(df, use_container_width=True)

    else:
        existing = [c for c in df.columns if re.fullmatch(r"cluster(_\d+)?", c)]
        if not existing:
            st.error("No cluster column found. Please run k-means above.")
            st.stop()
        cluster_col = st.sidebar.selectbox("Cluster column", existing)
        df[cluster_col] = df[cluster_col].astype(int)

# ─── 4b) Build 1-based string labels for plotting only ─────────────────────────
label_col = f"{cluster_col}_label"
df[label_col] = (df[cluster_col] + 1).astype(str)
label_order = sorted(df[label_col].unique(), key=lambda x: int(x))

# ─── 4c) Derive k_label for titles ────────────────────────────────────────────
if st.session_state.did_cluster and "n_clusters" in locals():
    k_label = n_clusters
else:
    parts = str(cluster_col).split("_")
    k_label = parts[-1] if parts[-1].isdigit() else ""

# ─── 5) PCA scores & loadings ───────────────────────────────────────────────────
pcs_in_file = [c for c in df.columns if re.fullmatch(r"(?i)PC[0-9]+", c)]
has_scores = len(pcs_in_file) >= 2

if not has_scores:
    pca_res = compute_pca_summary(df=df, hue_column=cluster_col)
    scores_df = pca_res["scores"]
    loadings = pca_res["loadings"]
    df = pd.concat([df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)
    pcs_in_file = scores_df.columns.tolist()
else:
    loadings = None

# ─── 6) Hover setup ────────────────────────────────────────────────────────────
hover_cols = [label_col] + [f"PC{i}" for i in (1, 2, 3) if f"PC{i}" in df.columns]
hover_template = "Cluster = %{customdata[0]}"
for idx, pc in enumerate(hover_cols[1:], start=1):
    hover_template += f"<br>{pc} = %{{customdata[{idx}]:.3f}}"

# ─── 7) Controls: 2D/3D & axes ─────────────────────────────────────────────────
dim_options = ["2D"] + (["3D"] if len(pcs_in_file) >= 3 else [])
plot_dim = st.sidebar.selectbox("Plot dimension", dim_options)

pc_x = st.sidebar.selectbox("X-Axis Principal Component", pcs_in_file, index=0)
pc_y = st.sidebar.selectbox(
    "Y-Axis Principal Component", [p for p in pcs_in_file if p != pc_x], index=0
)
pc_z = None
if plot_dim == "3D":
    pc_z = st.sidebar.selectbox(
        "Z-Axis Principal Component", [p for p in pcs_in_file if p not in (pc_x, pc_y)], index=0
    )

# ─── 8) Loading-vector scale ───────────────────────────────────────────────────
scale = st.sidebar.slider(
    "Loading Vector Scale",
    min_value=0.1,
    max_value=5.0,
    value=0.7,
    step=0.05,
    help="Proportion of PC-axis span for vector length",
)

# ─── 9) Header ────────────────────────────────────────────────────────────────
st.title("K-Means Clustering Dashboard")
st.markdown(f"**Dataset:** `{dataset_label}` — {df.shape[0]:,} rows, {df.shape[1]} cols")

# ─── 10) Plot ─────────────────────────────────────────────────────────────────
common = dict(
    color=label_col,
    category_orders={label_col: label_order},
    custom_data=hover_cols,
)

if plot_dim == "2D":
    fig = px.scatter(
        df,
        x=pc_x,
        y=pc_y,
        title=f"PCA Biplot Using {k_label} Clusters",
        hover_data=None,
        **common,
        width=900,
        height=900,
    )
    fig.update_traces(hovertemplate=hover_template, selector=dict(mode="markers"))
    # … loading-vector code unchanged …

    st.plotly_chart(fig, use_container_width=True)

else:
    fig3d = px.scatter_3d(
        df,
        x=pc_x,
        y=pc_y,
        z=pc_z,
        title=f"3D PCA Biplot Using {k_label} Clusters",
        hover_data=None,
        **common,
        width=1000,
        height=1000,
    )
    fig3d.update_traces(hovertemplate=hover_template, selector=dict(mode="markers"))
    # … 3D loading-vector code unchanged …

    st.plotly_chart(fig3d, use_container_width=True)