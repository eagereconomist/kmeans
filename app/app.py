import re

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from tennis_racquet_analysis.modeling.kmeans_utils import fit_kmeans
from tennis_racquet_analysis.preprocessing_utils import compute_pca_summary
from tennis_racquet_analysis.evaluation_utils import (
    compute_inertia_scores,
    compute_silhouette_scores,
)

# ─── 1) Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="K-Means Clustering Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("K-Means Clustering Dashboard")

# ─── RESTART BUTTON ───────────────────────────────────────────────────────────
if st.sidebar.button("Restart"):
    st.session_state.uploader_count = st.session_state.get("uploader_count", 0) + 1
    for key in (
        "did_cluster",
        "df",
        "cluster_col",
        "color_col",
        "cluster_order",
        "pve",
        "cpve",
        "loadings",
        "last_upload",
        "last_uploaded_name",
    ):
        st.session_state.pop(key, None)
    st.rerun()

st.sidebar.title("Dashboard Settings")

# ─── 2) Upload ─────────────────────────────────────────────────────────────────
uploader_key = f"uploader_{st.session_state.get('uploader_count', 0)}"
uploaded = st.sidebar.file_uploader(
    "Upload your own CSV", type="csv", key=uploader_key, help="Choose any local CSV to visualize"
)
if not uploaded:
    st.error("Please upload a CSV file to visualize.")
    st.stop()

# ─── Reset clustering state on new upload ──────────────────────────────────────
if st.session_state.get("last_upload") != uploaded.name:
    st.session_state.last_upload = uploaded.name
    for key in (
        "did_cluster",
        "df",
        "cluster_col",
        "color_col",
        "cluster_order",
        "pve",
        "cpve",
        "loadings",
    ):
        st.session_state.pop(key, None)

# ─── Safe CSV parsing ──────────────────────────────────────────────────────────
try:
    raw_df = pd.read_csv(uploaded)
except Exception:
    st.error("Error parsing the CSV. Please ensure it's well-formed.")
    st.stop()

if st.session_state.get("last_uploaded_name") != uploaded.name:
    st.session_state.did_cluster = False
    st.session_state.last_uploaded_name = uploaded.name

# ─── Input validation ───────────────────────────────────────────────────────────
if raw_df.isnull().any().any():
    st.error("Error: Dataset contains missing values. Clean and re-upload.")
    st.stop()

numeric_cols = raw_df.select_dtypes(include="number").columns.tolist()
has_cluster = any(re.search(r"cluster", c, re.I) for c in raw_df.columns)
if not has_cluster and len(numeric_cols) < 2:
    st.error("Need at least two numeric feature columns for k-means.")
    st.stop()

# ─── Pre-clustered detection ───────────────────────────────────────────────────
initial = [c for c in raw_df.columns if re.search(r"cluster", c, re.I)]
if initial:
    feature_cols = [c for c in numeric_cols if c not in initial]
    if len(feature_cols) < 2:
        st.error("Pre-clustered file must include ≥2 numeric feature columns aside from cluster.")
        st.stop()
    df_pc = raw_df.copy()
    orig = df_pc[initial[0]].astype(int)
    labels = orig - orig.min() + 1
    df_pc["cluster_label"] = labels.astype(str)
    st.session_state.df = df_pc
    st.session_state.cluster_col = initial[0]
    st.session_state.color_col = "cluster_label"
    st.session_state.cluster_order = [str(i) for i in sorted(labels.unique())]
    st.session_state.did_cluster = True

# ─── Detect pure PCA-scores file ────────────────────────────────────────────────
imported_pcs = [c for c in raw_df.columns if re.fullmatch(r"(?i)PC\d+", c)]
cluster_cols = [c for c in raw_df.columns if re.search(r"cluster", c, re.I)]
is_pca_scores_file = bool(imported_pcs) and set(raw_df.columns) <= set(imported_pcs + cluster_cols)

# ─── Display imported data ─────────────────────────────────────────────────────
st.markdown("## Imported Data")
dataset_placeholder = st.empty()
table_placeholder = st.empty()


def show_dataset(df: pd.DataFrame):
    dataset_placeholder.markdown(
        f"**Dataset:** `{uploaded.name}` — {df.shape[0]} rows, {df.shape[1]} cols"
    )
    table_placeholder.dataframe(df, use_container_width=True)


show_dataset(raw_df)

# ─── Cluster Diagnostics ──────────────────────────────────────────────────────
st.sidebar.header("Cluster Diagnostics")
max_k = st.sidebar.slider("Max Clusters (Diagnostics)", 3, 20, 10)
show_diag_data = st.sidebar.checkbox("Show Diagnostics Table", value=False)
show_inertia = st.sidebar.checkbox("Show Scree Plot", value=False)
show_silhouette = st.sidebar.checkbox("Show Silhouette Plot", value=False)

# ─── Model Settings ────────────────────────────────────────────────────────────
st.sidebar.header("Model Settings")
n_clusters = st.sidebar.slider("Number of clusters", 2, 20, 3)
n_init = st.sidebar.number_input("n_init (k-means)", min_value=1, value=50)
algo = st.sidebar.selectbox("Algorithm Method", ["lloyd", "elkan"])
init = st.sidebar.selectbox("Init Method", ["k-means++", "random"])
use_seed = st.sidebar.checkbox("Specify Random Seed", value=False)
seed = st.sidebar.number_input("Random seed", min_value=0, value=42) if use_seed else None

# ─── Compute diagnostics ───────────────────────────────────────────────────────
ks = list(range(1, max_k + 1))
inert_df = compute_inertia_scores(
    df=raw_df,
    k_range=ks,
    feature_columns=numeric_cols,
    init=init,
    n_init=n_init,
    random_state=seed,
    algorithm=algo,
)
inert_df["k"] = inert_df["k"].astype(int)

ks_sil = [k for k in ks if k >= 2]
if ks_sil:
    sil_df = (
        compute_silhouette_scores(
            df=raw_df,
            k_values=ks_sil,
            feature_columns=numeric_cols,
            init=init,
            n_init=n_init,
            random_state=seed,
            algorithm=algo,
        )
        .rename(columns={"n_clusters": "k"})
        .assign(k=lambda d: d["k"].astype(int))
        .set_index("k")
    )
else:
    sil_df = pd.DataFrame(columns=["silhouette_score"])
sil_ser = sil_df["silhouette_score"].reindex(ks)

if show_diag_data:
    diag_df = pd.DataFrame(
        {
            "k": ks,
            "inertia": inert_df.set_index("k").reindex(ks)["inertia"].tolist(),
            "silhouette": sil_ser.tolist(),
        }
    )
    st.markdown("#### Cluster Diagnostics Data")
    st.dataframe(diag_df.style.hide(axis="index"))

if show_inertia:
    st.markdown("### Inertia vs. k")
    fig_i = px.line(inert_df, x="k", y="inertia", markers=True)
    fig_i.update_xaxes(dtick=1, tickformat="d")
    st.plotly_chart(fig_i, use_container_width=True)

if show_silhouette:
    st.markdown("### Silhouette Score vs. k")
    sil_plot_df = sil_ser.reset_index().rename(
        columns={"index": "k", "silhouette_score": "silhouette_score"}
    )
    sil_plot_df = sil_plot_df[sil_plot_df["k"] >= 2]
    fig_s = px.line(sil_plot_df, x="k", y="silhouette_score", markers=True)
    fig_s.update_xaxes(tick0=2, dtick=1, tickformat="d")
    st.plotly_chart(fig_s, use_container_width=True)

# ─── Run or Re-run K-Means ──────────────────────────────────────────────────────
if not initial:
    if st.sidebar.button("Run K-Means"):
        df_clustered = fit_kmeans(
            raw_df.copy(),
            k=n_clusters,
            feature_columns=None,
            init=init,
            n_init=n_init,
            random_state=seed,
            algorithm=algo,
            label_column="cluster",
        )
        col = f"cluster_{n_clusters}"
        df_clustered[col] = df_clustered[col].astype(int)
        df_clustered["cluster_label"] = (df_clustered[col] + 1).astype(str)

        st.session_state.df = df_clustered
        st.session_state.cluster_col = col
        st.session_state.color_col = "cluster_label"
        st.session_state.cluster_order = [
            str(i) for i in sorted(df_clustered["cluster_label"].astype(int).unique())
        ]
        st.session_state.did_cluster = True

        show_dataset(df_clustered.drop(columns=["cluster_label"]))

# ─── Determine df & cluster_col ────────────────────────────────────────────────
if st.session_state.get("did_cluster", False):
    df = st.session_state.df.copy()
    cluster_col = st.session_state.cluster_col
else:
    df = raw_df.copy()
    cluster_col = None

# ─── PCA & Biplot (only after clustering) ──────────────────────────────────────
if not st.session_state.get("did_cluster", False):
    st.warning("Please run K-Means to view the PCA Biplot.")
else:
    if is_pca_scores_file:
        scores = raw_df[imported_pcs].copy()
        loadings = None
        variances = scores.var(ddof=0)
        pve = variances / variances.sum()
        cpve = pve.cumsum()
    else:
        pca = compute_pca_summary(df=df, hue_column=cluster_col)
        scores, loadings, pve, cpve = (
            pca["scores"],
            pca["loadings"],
            pca["pve"],
            pca["cpve"],
        )

    # combine scores
    old_pcs = [c for c in df.columns if re.fullmatch(r"(?i)PC\d+", c)]
    if old_pcs:
        df = df.drop(columns=old_pcs)
    df = pd.concat([df.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
    pcs = scores.columns.tolist()

    hover_cols = [st.session_state.get("color_col")] + pcs[:3]
    hover_template = "Cluster = %{customdata[0]}"
    for i, pc in enumerate(pcs[:3], 1):
        hover_template += f"<br>{pc} = %{{customdata[{i}]:.3f}}"

    st.sidebar.header("PCA Output Options")
    if is_pca_scores_file:
        show_pve = st.sidebar.checkbox("Show Proportional Variance Explained", value=False)
        show_cpve = st.sidebar.checkbox("Show Cumulative Variance Explained", value=False)
        show_scores = False
        show_loadings = False
    else:
        show_scores = st.sidebar.checkbox("Show Principal Component Scores", value=False)
        show_loadings = st.sidebar.checkbox("Show Principal Component Loadings", value=False)
        show_pve = st.sidebar.checkbox("Show Proportional Variance Explained", value=False)
        show_cpve = st.sidebar.checkbox("Show Cumulative Variance Explained", value=False)

    dim = st.sidebar.selectbox("Plot dimension", ["2D"] + (["3D"] if len(pcs) >= 3 else []))
    pc_x = st.sidebar.selectbox("X-Axis Principal Component", pcs, index=0)
    pc_y = st.sidebar.selectbox(
        "Y-Axis Principal Component", [p for p in pcs if p != pc_x], index=0
    )
    pc_z = None
    if dim == "3D":
        pc_z = st.sidebar.selectbox(
            "Z-Axis Principal Component", [p for p in pcs if p not in (pc_x, pc_y)], index=0
        )

    if not is_pca_scores_file:
        scale = st.sidebar.slider("Loading Vector Scale", 0.1, 5.0, 0.7, step=0.05)
    else:
        scale = None

    x_label = pc_x if pve.get(pc_x) is None else f"{pc_x} ({pve[pc_x]:.1%})"
    y_label = pc_y if pve.get(pc_y) is None else f"{pc_y} ({pve[pc_y]:.1%})"
    if dim == "3D":
        z_label = pc_z if pve.get(pc_z) is None else f"{pc_z} ({pve[pc_z]:.1%})"

    if dim == "2D" and len(pcs) < 2:
        st.error("At least 2 PCs required for a 2D biplot.")
        st.stop()
    if dim == "3D" and len(pcs) < 3:
        st.error("At least 3 PCs required for a 3D biplot.")
        st.stop()

    common = {
        "color": st.session_state["color_col"],
        "category_orders": {st.session_state["color_col"]: st.session_state["cluster_order"]},
        "custom_data": hover_cols,
    }

    # ─── 2D biplot ───────────────────────────────────────────────────────────────
    if dim == "2D":
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
        fig.update_traces(hovertemplate=hover_template, selector=dict(mode="markers"))
        fig.update_layout(title_font_size=24, xaxis_title=x_label, yaxis_title=y_label)
        fig.update_xaxes(title_font_size=17, tickfont_size=14)
        fig.update_yaxes(title_font_size=17, tickfont_size=14)

        if scale is not None:
            span_x = df[pc_x].max() - df[pc_x].min()
            span_y = df[pc_y].max() - df[pc_y].min()
            vec = min(span_x, span_y) * scale
            for feat in loadings.columns:
                x_end = loadings.at[pc_x, feat] * vec
                y_end = loadings.at[pc_y, feat] * vec
                fig.add_shape(
                    {
                        "type": "line",
                        "x0": 0,
                        "y0": 0,
                        "x1": x_end,
                        "y1": y_end,
                        "xref": "x",
                        "yref": "y",
                        "line": {"color": "grey", "width": 2},
                    }
                )
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
                    arrowhead=4,
                    arrowcolor="grey",
                    arrowwidth=2,
                    arrowsize=1,
                )
                fig.add_annotation(
                    x=x_end * 1.05,
                    y=y_end * 1.05,
                    showarrow=False,
                    text=feat,
                    font=dict(size=12, color="grey"),
                )
        st.plotly_chart(fig, use_container_width=True)

    # ─── 3D biplot ───────────────────────────────────────────────────────────────
    else:
        fig3d = px.scatter_3d(
            df,
            x=pc_x,
            y=pc_y,
            z=pc_z,
            title=f"PCA Biplot Using {cluster_col.split('_')[-1]} Clusters",
            hover_data=None,
            **common,
            width=1000,
            height=1000,
        )
        fig3d.update_traces(hovertemplate=hover_template, selector=dict(mode="markers"))
        fig3d.update_layout(
            title_font_size=24,
            scene=dict(
                xaxis_title=x_label,
                yaxis_title=y_label,
                zaxis_title=z_label,
                xaxis=dict(title_font_size=17, tickfont=dict(size=14)),
                yaxis=dict(title_font_size=17, tickfont=dict(size=14)),
                zaxis=dict(title_font_size=17, tickfont=dict(size=14)),
            ),
        )

        if scale is not None:
            spans = [df[c].max() - df[c].min() for c in (pc_x, pc_y, pc_z)]
            vec = min(spans) * scale
            frac = 0.1
            for feat in loadings.columns:
                x_e = loadings.at[pc_x, feat] * vec
                y_e = loadings.at[pc_y, feat] * vec
                z_e = loadings.at[pc_z, feat] * vec
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
                        sizeref=vec * frac,
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

    if show_scores:
        st.markdown("### PCA Scores")
        st.dataframe(scores, use_container_width=True)
    if show_loadings:
        st.markdown("### PCA Loadings")
        st.dataframe(loadings.T, use_container_width=True)
    if show_pve:
        st.markdown("### Percentage of Variance Explained")
        st.line_chart(pve)
    if show_cpve:
        st.markdown("### Cumulative Variance Explained")
        st.line_chart(cpve)

# ─── Cluster Profiling ─────────────────────────────────────────────────────────
st.sidebar.header("Cluster Profiling")
raw_prof = st.sidebar.file_uploader("Raw data (pre-standardized)", type="csv", key="prof_raw")
clust_prof = st.sidebar.file_uploader("Cluster results CSV", type="csv", key="prof_clust")

if raw_prof and clust_prof:
    raw_df = pd.read_csv(raw_prof)
    clust_df = pd.read_csv(clust_prof)

    # assign unique_id
    raw_df["unique_id"] = raw_df.index
    clust_df["unique_id"] = clust_df.index

    # only columns matching "cluster" are valid here
    cluster_opts = [c for c in clust_df.columns if re.search(r"cluster", c, re.I)]
    if not cluster_opts:
        st.error("No column matching 'cluster' found in your results CSV.")
        st.stop()

    # default to the one we used in-session if available
    default = st.session_state.get("cluster_col")
    prof_col = st.sidebar.selectbox(
        "Which column is your cluster ID?",
        cluster_opts,
        index=cluster_opts.index(default) if default in cluster_opts else 0,
    )

    # merge on unique_id, bring in only that one cluster column
    merged = (
        raw_df.merge(clust_df[["unique_id", prof_col]], on="unique_id")
        .rename(columns={prof_col: "cluster_label"})
        .drop(columns=["unique_id"])
    )

    # coerce to int and then to str for nice grouping
    merged["cluster_label"] = merged["cluster_label"].astype(int).astype(str)

    # 1) Cluster counts
    counts = (
        merged["cluster_label"]
        .value_counts()
        .sort_index(key=lambda idx: idx.astype(int))  # sort by numeric cluster ID
        .rename_axis("cluster_label")
        .reset_index(name="count")
    )
    st.markdown("### Cluster Counts")
    st.bar_chart(counts.set_index("cluster_label")["count"])

    # 2) Summary‐stat selector (mean always on)
    extra = st.sidebar.multiselect(
        "Additional stats to include", ["median", "min", "max"], default=[]
    )
    stats = ["mean"] + extra

    # inline helper, excludes PC* columns and the cluster_label itself
    from pandas.api import types as pd_types

    def _get_profiles(df, cluster_col, stats):
        feats = [
            c
            for c in df.columns
            if c != cluster_col
            and pd_types.is_numeric_dtype(df[c])
            and not re.fullmatch(r"PC\d+", c, flags=re.I)
        ]
        agg = df.groupby(cluster_col)[feats].agg(stats)
        agg.columns = [f"{feat}_{stat}" for feat, stat in agg.columns]
        return agg.reset_index()

    profiles = _get_profiles(merged, "cluster_label", stats)

    st.markdown("### Cluster Profiles")
    st.dataframe(profiles, use_container_width=True)
