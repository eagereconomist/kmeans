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
    "Upload your own CSV",
    type="csv",
    key=uploader_key,
    help="Choose any local CSV to visualize",
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
    st.error(
        "Error parsing the CSV file. Please ensure it's a well-formed CSV with consistent columns."
    )
    st.stop()

if st.session_state.get("last_uploaded_name") != uploaded.name:
    st.session_state.did_cluster = False
    st.session_state.last_uploaded_name = uploaded.name

# ─── Input validation ───────────────────────────────────────────────────────────
if raw_df.isnull().any().any():
    st.error(
        "Error: Uploaded dataset contains missing values. "
        "Please upload a cleaned dataset with no missing values."
    )
    st.stop()

numeric_cols = raw_df.select_dtypes(include="number").columns.tolist()
has_cluster = any(re.search(r"cluster", c, re.I) for c in raw_df.columns)
if not has_cluster and len(numeric_cols) < 2:
    st.error(
        "Error: This dataset isn't ready for k-means: it needs at least two numeric feature columns."
    )
    st.stop()

initial = [c for c in raw_df.columns if re.search(r"cluster", c, re.I)]
if initial:
    feature_cols = [c for c in numeric_cols if c not in initial]
    if len(feature_cols) < 2:
        st.error(
            "Your pre-clustered file must include at least two numeric feature columns "
            "in addition to the cluster column for PCA & visualization."
        )
        st.stop()
else:
    # no pre-existing cluster, must have ≥2 numeric cols
    if len(numeric_cols) < 2:
        st.error("We need at least two numeric feature columns for k-means.")
        st.stop()

# ─── Main app logic ────────────────────────────────────────────────────────────
st.markdown("## Imported Data")

# placeholders for the single dataset summary and the table
dataset_placeholder = st.empty()
table_placeholder = st.empty()


def show_dataset(df: pd.DataFrame):
    dataset_placeholder.markdown(
        f"**Dataset:** `{uploaded.name}` — {df.shape[0]} rows, {df.shape[1]} cols"
    )
    table_placeholder.dataframe(df, use_container_width=True)


# initial display
show_dataset(raw_df)

# ─── Branch: pre-clustered upload ─────────────────────────────────────────────
if initial:
    df = raw_df.copy()
    cluster_col = initial[0]
    df[cluster_col] = df[cluster_col].astype(int)
    df["cluster_label"] = (df[cluster_col] + 1).astype(str)

    st.session_state.color_col = "cluster_label"
    st.session_state.cluster_order = sorted(df["cluster_label"].unique(), key=int)
    st.session_state.did_cluster = False

    display_df = df.drop(columns=["cluster_label"])
    show_dataset(display_df)

# ─── Branch: fresh features ──────────────────────────────────────────────────
else:
    # before clustering
    if not st.session_state.get("did_cluster", False):
        show_dataset(raw_df)

        # ─── Model Settings ────────────────────────────────────────────
        st.sidebar.header("Model Settings")
        n_clusters = st.sidebar.slider("Number of clusters", 2, 15, 3)
        n_init = st.sidebar.number_input("n_init (k-means)", min_value=1, value=50)
        algo = st.sidebar.selectbox("Algorithm Method", ["lloyd", "elkan"])
        init = st.sidebar.selectbox("Init Method", ["k-means++", "random"])
        use_seed = st.sidebar.checkbox("Specify Random Seed", value=False)
        seed = st.sidebar.number_input("Random seed", min_value=0, value=42) if use_seed else None

        # ─── Cluster Diagnostics ────────────────────────────────────
        st.sidebar.header("Cluster Diagnostics")
        max_k = st.sidebar.slider("Max Clusters (Diagnostics)", 3, 15, 10)
        show_inertia = st.sidebar.checkbox("Show Scree Plot", value=False)
        show_silhouette = st.sidebar.checkbox("Show Silhouette Plot", value=False)
        show_diag_data = st.sidebar.checkbox("Show Diagnostics Table", value=False)

        if show_inertia or show_silhouette or show_diag_data:
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

        if st.sidebar.button("Run K-Means"):
            df = fit_kmeans(
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
            df[col] = df[col].astype(int)
            df["cluster_label"] = (df[col] + 1).astype(str)

            st.session_state.cluster_col = col
            st.session_state.color_col = "cluster_label"
            st.session_state.cluster_order = sorted(df["cluster_label"].unique(), key=int)
            st.session_state.did_cluster = True
            st.session_state.df = df

            display_df = df.drop(columns=["cluster_label"])
            show_dataset(display_df)
        else:
            st.info("Click **Run K-Means** to continue.")
            st.stop()

    # after clustering (reload)
    else:
        df = st.session_state.df
        display_df = df.drop(columns=["cluster_label"])
        show_dataset(display_df)

# ─── PCA & Biplot ────────────────────────────────────────────────────────────
cluster_col = st.session_state.get("cluster_col", None)
k_label = str(cluster_col).split("_")[-1] if cluster_col else ""

pca = compute_pca_summary(df=df, hue_column=cluster_col)
scores, loadings, pve, cpve = pca["scores"], pca["loadings"], pca["pve"], pca["cpve"]

imported_pcs = [c for c in raw_df.columns if re.fullmatch(r"(?i)PC\\d+", c)]
if imported_pcs:
    loadings = None

old_pcs = [c for c in df.columns if re.fullmatch(r"(?i)PC\\d+", c)]
if old_pcs:
    df = df.drop(columns=old_pcs)
df = pd.concat([df.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
pcs = scores.columns.tolist()

hover_cols = [st.session_state.get("color_col")] + pcs[:3]
hover_template = "Cluster = %{customdata[0]}"
for i, pc in enumerate(hover_cols[1:], 1):
    hover_template += f"<br>{pc} = %{{customdata[{i}]:.3f}}"

st.sidebar.header("PCA Output Options")
show_scores = st.sidebar.checkbox("Show Principal Component Scores", value=False)
show_loadings = st.sidebar.checkbox("Show Principal Component Loadings", value=False)
show_pve = st.sidebar.checkbox("Show Proportional Variance Explained", value=False)
show_cpve = st.sidebar.checkbox("Show Cumulative Variance Explained", value=False)

dim = st.sidebar.selectbox("Plot dimension", ["2D"] + (["3D"] if len(pcs) >= 3 else []))
pc_x = st.sidebar.selectbox("X-Axis Principal Component", pcs, index=0)
pc_y = st.sidebar.selectbox("Y-Axis Principal Component", [p for p in pcs if p != pc_x], index=0)
pc_z = None
if dim == "3D":
    pc_z = st.sidebar.selectbox(
        "Z-Axis Principal Component", [p for p in pcs if p not in (pc_x, pc_y)], index=0
    )

if dim == "2D" and len(pcs) < 2:
    st.error("At least 2 PCs required for a 2D biplot.")
    st.stop()
if dim == "3D" and len(pcs) < 3:
    st.error("At least 3 PCs required for a 3D biplot.")
    st.stop()

x_label = f"{pc_x} ({pve[pc_x]:.1%})"
y_label = f"{pc_y} ({pve[pc_y]:.1%})"
if dim == "3D":
    z_label = f"{pc_z} ({pve[pc_z]:.1%})"

scale = (
    st.sidebar.slider("Loading Vector Scale", 0.1, 5.0, 0.7, step=0.05)
    if loadings is not None
    else None
)

common = dict(
    color=st.session_state.get("color_col"),
    category_orders={st.session_state.get("color_col"): st.session_state.get("cluster_order", [])},
    custom_data=hover_cols,
)

if dim == "2D":
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
    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)

    if loadings is not None:
        span_x = df[pc_x].max() - df[pc_x].min()
        span_y = df[pc_y].max() - df[pc_y].min()
        vec = min(span_x, span_y) * scale
        for feat in loadings.columns:
            x_end = loadings.at[pc_x, feat] * vec
            y_end = loadings.at[pc_y, feat] * vec
            fig.add_shape(
                dict(
                    type="line",
                    x0=0,
                    y0=0,
                    x1=x_end,
                    y1=y_end,
                    xref="x",
                    yref="y",
                    line=dict(color="grey", width=2),
                )
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
    fig3d.update_layout(
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label,
        )
    )

    if loadings is not None:
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

# ─── Optional PCA tables & charts ─────────────────────────────────────────
if show_scores:
    st.markdown("### PCA Scores")
    st.dataframe(scores)
if show_loadings and loadings is not None:
    st.markdown("### PCA Loadings")
    st.dataframe(loadings.T)
if show_pve:
    st.markdown("### Percentage of Variance Explained")
    st.line_chart(pve)
if show_cpve:
    st.markdown("### Cumulative Variance Explained")
    st.line_chart(cpve)
