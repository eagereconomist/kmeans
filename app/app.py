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

# Remember if we've already run k-means in this session
if "did_cluster" not in st.session_state:
    st.session_state.did_cluster = False

# ─── 2) Upload CSV ─────────────────────────────────────────────────────────────
uploaded = st.sidebar.file_uploader(
    "Upload your own CSV", type="csv", help="Choose any local CSV to visualize"
)
if not uploaded:
    st.error("Please upload a CSV file to visualize.")
    st.stop()

# ─── 2b) Restore prior clustered DF or read afresh ─────────────────────────────
if st.session_state.did_cluster and "df_clustered" in st.session_state:
    df = st.session_state.df_clustered.copy()
    dataset_label = st.session_state.dataset_label
else:
    df = pd.read_csv(uploaded)
    dataset_label = uploaded.name

# ─── 2c) Pre-existing cluster column? ─────────────────────────────────────────
initial = [c for c in df.columns if re.search(r"cluster(_\d+)?", c, re.I)]
if initial and not st.session_state.did_cluster:
    # let the user pick the numeric, zero-based column
    numeric_col = st.sidebar.selectbox("Cluster column", initial)
    df[numeric_col] = df[numeric_col].astype(int)

    st.error(
        "Input file already contains cluster column(s); using "
        f"`{numeric_col}` for plotting and comparison"
    )
    st.subheader("Imported Data (with existing clusters)")
    st.dataframe(df, use_container_width=True)

else:
    # ─── 3) Model Settings ───────────────────────────────────────────────────────
    st.sidebar.header("Model Settings")
    n_clusters = st.sidebar.slider("Number of clusters", 2, 15, 3)
    n_init = st.sidebar.number_input("n_init (k-means)", 1, 100, 50)
    algo = st.sidebar.selectbox("Algorithm Method", ["lloyd", "elkan"])
    init = st.sidebar.selectbox("Init Method", ["k-means++", "random"])
    use_seed = st.sidebar.checkbox("Specify Random Seed", False)
    seed = st.sidebar.number_input("Random seed", 0, 99999, 42) if use_seed else None
    go_btn = st.sidebar.button("Run K-Means")

    if go_btn:
        # run k-means, this creates a column "cluster_{n_clusters}"
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
        numeric_col = f"cluster_{n_clusters}"
        df[numeric_col] = df[numeric_col].astype(int)

        st.subheader("Clustered Data")
        st.dataframe(df, use_container_width=True)

        # cache for reruns
        st.session_state.df_clustered = df.copy()
        st.session_state.dataset_label = dataset_label

    else:
        # pick from any existing "cluster" or "cluster_k"
        existing = [c for c in df.columns if re.fullmatch(r"cluster(_\d+)?", c)]
        if not existing:
            st.error("No cluster column found. Please run k-means above.")
            st.stop()
        numeric_col = st.sidebar.selectbox("Cluster column", existing)
        df[numeric_col] = df[numeric_col].astype(int)

        # cache so dimension switch won’t reset
        st.session_state.did_cluster = True
        st.session_state.df_clustered = df.copy()
        st.session_state.dataset_label = dataset_label

# ─── 4) Create display column at +1 for legend only ───────────────────────────
df["cluster_display"] = (df[numeric_col] + 1).astype(str)
cluster_col = "cluster_display"
cluster_order = sorted(df[cluster_col].unique(), key=lambda x: int(x))

# ─── 5) k_label for titles ─────────────────────────────────────────────────────
parts = numeric_col.split("_")
k_label = parts[-1] if parts[-1].isdigit() else ""

# ─── 6) PCA scores & loadings ───────────────────────────────────────────────────
pcs = [c for c in df.columns if re.fullmatch(r"(?i)PC[0-9]+", c)]
if len(pcs) < 2:
    res = compute_pca_summary(df=df, hue_column=cluster_col)
    df = pd.concat([df, res["scores"]], axis=1)
    loadings = res["loadings"]
    pcs = res["scores"].columns.tolist()
else:
    loadings = None

# ─── 7) Hovertemplate ──────────────────────────────────────────────────────────
hover_cols = [cluster_col] + pcs[:3]
ht = "Cluster = %{customdata[0]}"
for i, pc in enumerate(hover_cols[1:], 1):
    ht += f"<br>{pc} = %{{customdata[{i}]:.3f}}"

# ─── 8) 2D/3D selector & axes ───────────────────────────────────────────────────
dims = ["2D"] + (["3D"] if len(pcs) >= 3 else [])
plot_dim = st.sidebar.selectbox("Plot dimension", dims)

pc_x = st.sidebar.selectbox("X-Axis Principal Component", pcs, index=0)
pc_y = st.sidebar.selectbox("Y-Axis Principal Component", [p for p in pcs if p != pc_x], index=0)
pc_z = None
if plot_dim == "3D":
    pc_z = st.sidebar.selectbox("Z-Axis Principal Component", [p for p in pcs if p not in (pc_x, pc_y)], index=0)

# ─── 9) Loading-vector scale ───────────────────────────────────────────────────
scale = st.sidebar.slider("Loading Vector Scale", 0.1, 5.0, 0.7, 0.05)

# ─── 10) Header & summary ───────────────────────────────────────────────────────
st.title("K-Means Clustering Dashboard")
st.markdown(f"**Dataset:** `{dataset_label}` — {df.shape[0]:,} rows, {df.shape[1]} cols")

# ─── 11) Plot ──────────────────────────────────────────────────────────────────
common = dict(
    color=cluster_col,
    category_orders={cluster_col: cluster_order},
    custom_data=hover_cols,
)

if plot_dim == "2D":
    fig = px.scatter(
        df, x=pc_x, y=pc_y,
        title=f"PCA Biplot Using {k_label} Clusters",
        hover_data=None, **common, width=900, height=900
    )
    fig.update_traces(hovertemplate=ht, selector=dict(mode="markers"))
    # … add your 2D arrows here as before …
    st.plotly_chart(fig, use_container_width=True)

else:
    fig3d = px.scatter_3d(
        df, x=pc_x, y=pc_y, z=pc_z,
        title=f"3D PCA Biplot Using {k_label} Clusters",
        hover_data=None, **common, width=1000, height=1000
    )
    fig3d.update_traces(hovertemplate=ht, selector=dict(mode="markers"))
    # … add your 3D arrows here as before …
    st.plotly_chart(fig3d, use_container_width=True)