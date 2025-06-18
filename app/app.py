import re

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from tennis_racquet_analysis.modeling.kmeans_utils import fit_kmeans
from tennis_racquet_analysis.config import PROCESSED_DATA_DIR
from tennis_racquet_analysis.preprocessing_utils import (
    load_data,  # loads a CSV given a Path
    compute_pca_summary,  # returns {'scores': DataFrame, 'loadings': DataFrame}
)

# ─── 1) Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="K-Means Clustering Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.sidebar.title("Dashboard Settings")

# ─── 2) Load data ───────────────────────────────────────────────────────────────
uploaded = st.sidebar.file_uploader("Upload your own CSV", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    dataset_label = uploaded.name
else:
    processed = PROCESSED_DATA_DIR
    all_csvs = sorted(processed.rglob("*.csv"))
    choices = [str(p.relative_to(processed.parent)) for p in all_csvs]
    dataset_label = st.sidebar.selectbox("Or choose a project CSV", choices)
    df = load_data(processed.parent / dataset_label)

# ─── 3) Model Settings ─────────────────────────────────────────────────────────
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

# ─── 4) Run or pick clusters ────────────────────────────────────────────────────
if run_cluster:
    df = fit_kmeans(
        df,
        k=n_clusters,
        feature_columns=None,
        init=init_method,
        n_init=n_init,
        random_state=(random_seed if use_random_seed else None),
        algorithm=algo_method,
        label_column="cluster",
    )
    st.session_state.did_cluster = True

    cluster_col = f"cluster_{n_clusters}"
    cluster_order = sorted(df[cluster_col].unique(), key=lambda x: int(x))

    st.subheader("Clustered Data")
    st.dataframe(df)

else:
    existing = [c for c in df.columns if re.fullmatch(r"cluster(_\d+)?", c)]
    if not existing:
        st.error("No cluster column found. Please run k-means above.")
        st.stop()
    cluster_col = st.sidebar.selectbox("Cluster column", existing)
    cluster_order = sorted(df[cluster_col].unique(), key=lambda x: int(x))

# ─── 4b) Derive k_label for titles ─────────────────────────────────────────────
if st.session_state.did_cluster:
    k_label = n_clusters
elif cluster_col is not None:
    k_label = int(cluster_col.split("_")[-1])
else:
    k_label = ""

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
hover_cols = [cluster_col] + [f"PC{i}" for i in (1, 2, 3) if f"PC{i}" in df.columns]
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
    color=cluster_col, category_orders={cluster_col: cluster_order}, custom_data=hover_cols
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

    if loadings is not None:
        span_x = df[pc_x].max() - df[pc_x].min()
        span_y = df[pc_y].max() - df[pc_y].min()
        vec_scale = min(span_x, span_y) * scale
        for feat in loadings.columns:
            x_end = loadings.at[pc_x, feat] * vec_scale
            y_end = loadings.at[pc_y, feat] * vec_scale

            # shaft
            fig.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=x_end,
                y1=y_end,
                xref="x",
                yref="y",
                line=dict(color="grey", width=2),
            )
            # arrowhead
            fig.add_annotation(
                x=x_end,
                y=y_end,
                ax=0,
                ay=0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowcolor="grey",
                arrowwidth=2,
                arrowsize=2,
                text="",
            )
            # label
            fig.add_annotation(
                x=x_end * 1.05,
                y=y_end * 1.05,
                showarrow=False,
                text=feat,
                font=dict(size=12, color="grey"),
            )

    st.plotly_chart(fig, use_container_width=True)

else:  # 3D
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

    if loadings is not None:
        spans = [df[c].max() - df[c].min() for c in (pc_x, pc_y, pc_z)]
        vec_scale = min(spans) * scale
        frac = 0.1
        for feat in loadings.columns:
            x_e = loadings.at[pc_x, feat] * vec_scale
            y_e = loadings.at[pc_y, feat] * vec_scale
            z_e = loadings.at[pc_z, feat] * vec_scale
            x_s, y_s, z_s = x_e * (1 - frac), y_e * (1 - frac), z_e * (1 - frac)

            # shaft
            fig3d.add_trace(
                go.Scatter3d(
                    x=[0, x_s],
                    y=[0, y_s],
                    z=[0, z_s],
                    mode="lines",
                    line=dict(color="grey", width=2),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            # arrowhead cone
            fig3d.add_trace(
                go.Cone(
                    x=[x_s],
                    y=[y_s],
                    z=[z_s],
                    u=[x_e - x_s],
                    v=[y_e - y_s],
                    w=[z_e - z_s],
                    anchor="tail",
                    showscale=False,
                    colorscale=[[0, "grey"], [1, "grey"]],
                    sizemode="absolute",
                    sizeref=vec_scale * frac,
                )
            )
            # label
            fig3d.add_trace(
                go.Scatter3d(
                    x=[x_e],
                    y=[y_e],
                    z=[z_e],
                    mode="text",
                    text=[feat],
                    textfont=dict(size=12, color="grey"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    st.plotly_chart(fig3d, use_container_width=True)
