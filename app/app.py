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

# ─── Always show the raw imported data ────────────────────────────────────────
table = st.empty()
table.dataframe(df, use_container_width=True)

# ─── 2b) Pre-existing clusters? ───────────────────────────────────────────────
initial = [c for c in df.columns if re.search(r"cluster", c, re.I)]
if initial:
    st.session_state.did_cluster = False
    st.warning(f"File already has cluster column(s): {', '.join(initial)} (using first)…")
    cluster_col = initial[0]
    df[cluster_col] = df[cluster_col].astype(int)  # keep 0-based internally

else:
    # ─── 3) Model Settings ───────────────────────────────────────────────────
    st.sidebar.header("Model Settings")
    n_clusters = st.sidebar.slider("Number of clusters", 2, 15, 3)
    n_init = st.sidebar.number_input("n_init (k-means)", 1, 100, 50)
    algo = st.sidebar.selectbox("Algorithm", ["lloyd", "elkan"])
    init = st.sidebar.selectbox("Init method", ["k-means++", "random"])
    use_seed = st.sidebar.checkbox("Specify random seed")
    seed = st.sidebar.number_input("Random seed", 0, 9999, 42) if use_seed else None
    run = st.sidebar.button("Run K-Means")

    if run:
        df = fit_kmeans(
            df,
            k=n_clusters,
            feature_columns=None,
            init=init,
            n_init=n_init,
            random_state=seed,
            algorithm=algo,
            label_column="cluster",
        )
        st.session_state.did_cluster = True
        cluster_col = f"cluster_{n_clusters}"
        df[cluster_col] = df[cluster_col].astype(int)  # keep 0-based internally
    else:
        existing = [c for c in df.columns if re.fullmatch(r"cluster(_\d+)?", c, re.I)]
        if not existing:
            st.error("No cluster column found. Please run k-means above.")
            st.stop()
        cluster_col = existing[0]
        df[cluster_col] = df[cluster_col].astype(int)

# ─── 4) Create 1-based “plot” column for legend ────────────────────────────────
plot_col = f"{cluster_col}_plot"
df[plot_col] = (df[cluster_col] + 1).astype(str)
cluster_order = sorted(df[plot_col].unique(), key=lambda x: int(x))

# Update the single table in-place to include the cluster column
table.dataframe(df, use_container_width=True)

# ─── 5) PCA & loadings ─────────────────────────────────────────────────────────
pcs = [c for c in df.columns if re.fullmatch(r"(?i)PC[0-9]+", c)]
if len(pcs) < 2:
    pca = compute_pca_summary(df=df, hue_column=plot_col)
    scores, loadings = pca["scores"], pca["loadings"]
    df = pd.concat([df.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
    pcs = scores.columns.tolist()
else:
    loadings = None

# ─── 6) Hover setup ───────────────────────────────────────────────────────────
hover_cols = [plot_col] + pcs[:3]
template = "Cluster = %{customdata[0]}"
for i, pc in enumerate(hover_cols[1:], start=1):
    template += f"<br>{pc} = %{{customdata[{i}]:.3f}}"

# ─── 7) 2D/3D selector ──────────────────────────────────────────────────────────
dim_opts = ["2D"] + (["3D"] if len(pcs) >= 3 else [])
plot_dim = st.sidebar.selectbox("Plot dimension", dim_opts)
pc_x = st.sidebar.selectbox("X-Axis Principal Component", pcs, index=0)
pc_y = st.sidebar.selectbox("Y-Axis Principal Component", [p for p in pcs if p != pc_x], index=0)
pc_z = None
if plot_dim == "3D":
    pc_z = st.sidebar.selectbox(
        "Z-Axis Principal Component", [p for p in pcs if p not in (pc_x, pc_y)], index=0
    )

scale = st.sidebar.slider("Loading Vector Scale", 0.1, 5.0, 0.7, step=0.05)

st.title("K-Means Clustering Dashboard")
st.markdown(f"**Dataset:** `{dataset_label}` — {df.shape[0]:,} rows, {df.shape[1]} cols")

common = dict(
    color=plot_col,
    category_orders={plot_col: cluster_order},
    custom_data=hover_cols,
)

# ─── 8) 2D plot ────────────────────────────────────────────────────────────────
if plot_dim == "2D":
    fig = px.scatter(
        df,
        x=pc_x,
        y=pc_y,
        title=f"PCA Biplot Using {cluster_col.split('_')[-1]} Clusters",
        hover_data=None,
        **common,
        width=900,
        height=900,
    )
    fig.update_traces(hovertemplate=template, selector=dict(mode="markers"))

    if loadings is not None:
        span_x = df[pc_x].max() - df[pc_x].min()
        span_y = df[pc_y].max() - df[pc_y].min()
        vec_scale = min(span_x, span_y) * scale

        for feat in loadings.columns:
            x_end = loadings.at[pc_x, feat] * vec_scale
            y_end = loadings.at[pc_y, feat] * vec_scale

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
            fig.add_annotation(
                x=x_end,
                y=y_end,
                ax=x_end * 0.85,
                ay=y_end * 0.85,
                showarrow=True,
                arrowhead=3,
                arrowcolor="grey",
                arrowwidth=2,
                arrowsize=1,
                text="",
            )
            fig.add_annotation(
                x=x_end * 1.05,
                y=y_end * 1.05,
                showarrow=False,
                text=feat,
                font=dict(size=12, color="grey"),
            )

    st.plotly_chart(fig, use_container_width=True)

# ─── 9) 3D plot ────────────────────────────────────────────────────────────────
else:
    fig3d = px.scatter_3d(
        df,
        x=pc_x,
        y=pc_y,
        z=pc_z,
        title=f"3D PCA Biplot Using {cluster_col.split('_')[-1]} Clusters",
        hover_data=None,
        **common,
        width=1000,
        height=1000,
    )
    fig3d.update_traces(hovertemplate=template, selector=dict(mode="markers"))

    if loadings is not None:
        spans = [
            df[c].max() - df[c].min() for c in (pc_x, pc_y, pc_z)
        ]
        vec_scale = min(spans) * scale
        frac = 0.1

        for feat in loadings.columns:
            x_e = loadings.at[pc_x, feat] * vec_scale
            y_e = loadings.at[pc_y, feat] * vec_scale
            z_e = loadings.at[pc_z, feat] * vec_scale
            x_s, y_s, z_s = x_e * (1 - frac), y_e * (1 - frac), z_e * (1 - frac)

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