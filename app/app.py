import os
import io
import re

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from pandas.api import types as pd_types
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
    "Upload your own data",
    type=["csv", "txt", "xlsx", "xls"],
    key=uploader_key,
    help="Choose any local data file to begin",
)
if not uploaded:
    st.error("Please upload a data file to proceed.")
    st.stop()

# derive base name for downloads
base_name, ext = os.path.splitext(uploaded.name.lower())

# set download-format default to match uploaded file extension
_format_opts = ["csv", "txt", "xlsx", "xls"]
_default_fmt = ext.lstrip(".") if ext.lstrip(".") in _format_opts else "csv"

# ─── Download format selector ─────────────────────────────────────────────
download_format = st.sidebar.selectbox(
    "Download format",
    _format_opts,
    index=_format_opts.index(_default_fmt),
    help="Choose the file format for all downloads",
)


def make_download(df: pd.DataFrame, name: str, key: str):
    fname = f"{name}.{download_format}"
    if download_format == "csv":
        data = df.to_csv(index=False).encode("utf-8")
        mime = "text/csv"
    elif download_format == "txt":
        data = df.to_csv(sep=",", index=False).encode("utf-8")
        mime = "text/tab-separated-values"
    elif download_format == "xlsx":
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        data = buf.getvalue()
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        st.error(f"Unsupported format: {download_format}")
        return

    st.sidebar.download_button(
        f"Download (.{download_format})",
        data,
        file_name=fname,
        mime=mime,
        key=key,
    )


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

# ─── Safe file parsing ──────────────────────────────────────────────────────────
progress_bar = st.sidebar.progress(0, text="Reading file...")

try:
    if ext in (".csv", ".txt"):
        sep = ","
        raw_df = pd.read_csv(uploaded, sep=sep)
    elif ext in (".xlsx", ".xls"):
        raw_df = pd.read_excel(uploaded)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    progress_bar.progress(100, text="File successfully loaded!")

except Exception as e:
    st.error(f"Error reading `{uploaded.name}`: {e}")
    st.stop()

finally:
    progress_bar.empty()

# assign a unique ID for each row
raw_df["unique_id"] = raw_df.index.astype(str)

if st.session_state.get("last_uploaded_name") != uploaded.name:
    st.session_state.did_cluster = False
    st.session_state.last_uploaded_name = uploaded.name

# ─── Input validation ───────────────────────────────────────────────────────────
if raw_df.isnull().any().any():
    st.error("Error: Dataset contains missing values. Clean and re-upload.")
    st.stop()

numeric_cols = raw_df.select_dtypes(include="number").columns.tolist()
has_cluster = any(re.search(r"cluster", cluster, re.I) for cluster in raw_df.columns)
if not has_cluster and len(numeric_cols) < 2:
    st.error("Need at least two numeric feature columns for K-Means.")
    st.stop()

# ─── Pre-clustered detection ───────────────────────────────────────────────────
initial = [cluster for cluster in raw_df.columns if re.search(r"cluster", cluster, re.I)]
if initial:
    feature_cols = [cluster for cluster in numeric_cols if cluster not in initial]
    if len(feature_cols) < 2:
        st.error(
            "Pre-clustered file must include 2 or more numeric feature columns aside from the cluster column."
        )
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
imported_pcs = [column for column in raw_df.columns if re.fullmatch(r"(?i)PC\d+", column)]
cluster_cols = [column for column in raw_df.columns if re.search(r"cluster", column, re.I)]
cols_without_id = set(raw_df.columns) - {"unique_id"}
is_pca_scores_file = bool(imported_pcs) and cols_without_id <= set(imported_pcs + cluster_cols)

# ─── Detect PC-loadings file (rows == #PCs, but no other obs) ───────────────────
is_pca_loadings_file = is_pca_scores_file and raw_df.shape[0] == len(imported_pcs)

# ─── Flags for hiding controls ─────────────────────────────────────────────────

# disable settings only for pre-clustered imports (initial) or PC-loadings
show_model_settings = not initial and not is_pca_loadings_file

# keep diagnostics disabled only for pre-clustered or pure PC-scores files
show_diagnostics = not initial and not is_pca_scores_file

# ─── Display imported data ─────────────────────────────────────────────────────
st.markdown("## Imported Data")
dataset_placeholder = st.empty()
table_placeholder = st.empty()


def show_dataset(df: pd.DataFrame):
    dataset_placeholder.markdown(
        f"**Dataset:** `{uploaded.name}` — {df.shape[0]} rows, {df.shape[1]} cols"
    )
    table_placeholder.dataframe(df.set_index("unique_id"), use_container_width=True)


show_dataset(raw_df)

# Ensure model settings keys exist in session_state with defaults
for key, default in [
    ("n_clusters", 3),
    ("n_init", 50),
    ("algo", "lloyd"),
    ("init", "k-means++"),
    ("seed", 42),
]:
    st.session_state.setdefault(key, default)

# Pull the current settings into local vars so diagnostics can use them
n_clusters = st.session_state.n_clusters
n_init = st.session_state.n_init
algo = st.session_state.algo
init = st.session_state.init
seed = st.session_state.seed

# ─── Cluster Diagnostics ──────────────────────────────────────────────────────
# only show diagnostics when we have raw feature data (not pre-clustered/pure PC-loadings)
if not initial and not is_pca_loadings_file:
    st.sidebar.header("Cluster Diagnostics")
    max_k = st.sidebar.slider("Max Clusters (Diagnostics)", 3, 20, 10)
    show_diag_data = st.sidebar.checkbox("Show Diagnostics Table", value=False)
    show_inertia = st.sidebar.checkbox("Show Scree Plot", value=False)
    show_silhouette = st.sidebar.checkbox("Show Silhouette Plot", value=False)

    diagnostics_requested = show_diag_data or show_inertia or show_silhouette

    if diagnostics_requested:
        # use a dedicated progress bar
        diag_bar = st.sidebar.progress(0, text="Running Clustering Diagnostics...")
        try:
            ks = list(range(1, max_k + 1))

            # Inertia
            if show_diag_data or show_inertia:
                diag_bar.progress(10, text="Computing Inertia Scores...")
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
            else:
                inert_df = pd.DataFrame()

            # Silhouette
            if show_silhouette:
                ks_sil = [k for k in ks if k >= 2]
                if ks_sil:
                    diag_bar.progress(60, text="Computing Silhouette Scores...")
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
            else:
                sil_df = pd.DataFrame()
                sil_ser = pd.Series(dtype=float)

            diag_bar.progress(100, text="Diagnostics Complete!")

        except Exception as e:
            st.error(f"Could not compute diagnostics: {e}")
            inert_df = pd.DataFrame()
            sil_df = pd.DataFrame()
            sil_ser = pd.Series(dtype=float)

        finally:
            # guaranteed to clear the bar
            diag_bar.empty()

    # ─── Diagnostics Table ────────────────────────────────────────────────────────
    if show_diag_data:
        if inert_df.empty and sil_ser.empty:
            st.info("Select at least one plot or the diagnostics table above to generate metrics.")
        else:
            diag_df = pd.DataFrame()
            if not inert_df.empty:
                diag_df["k"] = inert_df["k"]
                diag_df["inertia"] = inert_df["inertia"]
            if not sil_ser.empty:
                diag_df["silhouette"] = sil_ser.values[: len(diag_df)]
            st.markdown("#### Cluster Diagnostics Data")
            st.dataframe(diag_df)

            # Download Cluster Diagnostics Data
            make_download(
                diag_df,
                f"{base_name}_cluster_diagnostics",
                f"download_cluster_diagnostics_{download_format}",
            )

    # ─── Inertia Plot ─────────────────────────────────────────────────────────────
    if show_inertia:
        st.markdown("### Inertia vs. k")
        fig_i = px.line(inert_df, x="k", y="inertia", markers=True)
        fig_i.update_xaxes(dtick=1, tickformat="d")
        st.plotly_chart(fig_i, use_container_width=True)

    # ─── Silhouette Plot ──────────────────────────────────────────────────────────
    if show_silhouette:
        st.markdown("### Silhouette Score vs. k")
        sil_plot_df = sil_ser.reset_index().rename(
            columns={"index": "k", "silhouette_score": "silhouette_score"}
        )
        sil_plot_df = sil_plot_df[sil_plot_df["k"] >= 2]
        fig_s = px.line(sil_plot_df, x="k", y="silhouette_score", markers=True)
        fig_s.update_xaxes(tick0=2, dtick=1, tickformat="d")
        st.plotly_chart(fig_s, use_container_width=True)

    else:
        # diagnostics hidden
        max_k = 10
        show_diag_data = False
        show_inertia = False
        show_silhouette = False
        inert_df = pd.DataFrame()
        sil_df = pd.DataFrame()
        sil_ser = pd.Series(dtype=float)


# ─── Model Settings ────────────────────────────────────────────────────────────
if show_model_settings:
    st.sidebar.header("Model Settings")
    st.sidebar.slider("Number of clusters", 2, 20, key="n_clusters")
    st.sidebar.number_input("n_init (k-means)", min_value=50, key="n_init")
    st.sidebar.selectbox(
        "Algorithm Method",
        ["lloyd", "elkan"],
        key="algo",
    )
    st.sidebar.selectbox(
        "Init Method",
        ["k-means++", "random"],
        key="init",
    )
    st.sidebar.number_input(
        "Random seed",
        min_value=0,
        value=st.session_state.seed,
        key="seed",
        help="Use this seed for reproducible clustering.",
    )


# ─── Run or Re-run K-Means ─────────────────────────────────────────────────────
if show_model_settings:
    if st.sidebar.button("Run K-Means"):
        progress_bar = st.sidebar.progress(0, text="Running K-Means Clustering...")
        try:
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
            progress_bar.progress(100, text="K-Means Completed!")
        except Exception:
            st.error(
                "An error occurred during K-Means clustering. Please verify your data and settings."
            )
        else:
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

# ─── Download Clustering Results ────────────────────────────────────────────────
if st.session_state.get("did_cluster", False):
    export_df = st.session_state.df.drop(columns=["cluster_label"])
    make_download(
        export_df,
        f"{base_name}_cluster_{st.session_state.n_clusters}",
        f"download_clustered_{st.session_state.n_clusters}_{download_format}",
    )

# ─── Determine df & cluster_col ────────────────────────────────────────────────
if st.session_state.get("did_cluster", False):
    df = st.session_state.df.copy()
    cluster_col = st.session_state.cluster_col
else:
    df = raw_df.copy()
    cluster_col = None

# ─── Always refresh the table with the current df ───────────────────────────────
# (drop the helper column if present)
to_show = df.drop(columns=["cluster_label"]) if "cluster_label" in df.columns else df
show_dataset(to_show)

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
        progress_bar = st.progress(0, text="Computing PCA Summary...")
        try:
            pca = compute_pca_summary(df=df, hue_column=cluster_col)
            scores, loadings, pve, cpve = (
                pca["scores"],
                pca["loadings"],
                pca["pve"],
                pca["cpve"],
            )
        except Exception as e:
            progress_bar.empty()
            st.error(f"PCA computation failed: {e}")
            st.stop()
        else:
            progress_bar.progress(100, text="PCA Computation Complete!")
        finally:
            progress_bar.empty()

    # combine scores with original df
    old_pcs = [column for column in df.columns if re.fullmatch(r"(?i)PC\d+", column)]
    if old_pcs:
        df = df.drop(columns=old_pcs)
    df = pd.concat([df.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
    pcs = scores.columns.tolist()

    hover_cols = ["unique_id", st.session_state.get("color_col")] + pcs[:3]
    hover_template = "Unique ID = %{customdata[0]}<br>Cluster = %{customdata[1]}"
    for i, pc in enumerate(pcs[:3], start=2):
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
        "Y-Axis Principal Component", [pc for pc in pcs if pc != pc_x], index=0
    )
    pc_z = None
    if dim == "3D":
        pc_z = st.sidebar.selectbox(
            "Z-Axis Principal Component", [pc for pc in pcs if pc not in (pc_x, pc_y)], index=0
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

    # 2D PC Biplot
    if dim == "2D":
        fig = px.scatter(
            df,
            x=pc_x,
            y=pc_y,
            title=f"Principal Component Biplot Using {cluster_col.split('_')[-1]} Clusters",
            hover_data=None,
            **common,
            width=900,
            height=900,
        )
        fig.update_traces(hovertemplate=hover_template, selector=dict(mode="markers"))
        fig.update_layout(title_font_size=27, xaxis_title=x_label, yaxis_title=y_label)
        fig.update_xaxes(title_font_size=17, tickfont_size=14)
        fig.update_yaxes(title_font_size=17, tickfont_size=14)

        if show_scores:
            make_download(
                scores, f"{base_name}_pc_scores", f"download_pc_scores_{download_format}"
            )

        if show_loadings:
            loadings_df = loadings.T.reset_index().rename(columns={"index": "component"})
            make_download(
                loadings_df, f"{base_name}_pc_loadings", f"download_loadings_{download_format}"
            )

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
                    },
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

    # 3D PC Biplot
    else:
        fig3d = px.scatter_3d(
            df,
            x=pc_x,
            y=pc_y,
            z=pc_z,
            title=f"Principal Component Biplot Using {cluster_col.split('_')[-1]} Clusters",
            hover_data=None,
            **common,
            width=1000,
            height=1000,
        )
        fig3d.update_traces(hovertemplate=hover_template, selector=dict(mode="markers"))
        fig3d.update_layout(
            title_font_size=27,
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
            spans = [df[column].max() - df[column].min() for column in (pc_x, pc_y, pc_z)]
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
        st.markdown("### Principal Component Scores")
        st.dataframe(scores, use_container_width=True)
    if show_loadings:
        st.markdown("### Principal Component Loadings")
        st.dataframe(loadings.T, use_container_width=True)
    if show_pve:
        st.markdown("### Percentage of Variance Explained")
        st.line_chart(pve)
    if show_cpve:
        st.markdown("### Cumulative Variance Explained")
        st.line_chart(cpve)

# ─── Cluster Profiling ─────────────────────────────────────────────────────────
st.sidebar.header("Cluster Profiling")

# allow same file types as main uploader
upload_types = ["csv", "txt", "xlsx", "xls"]
raw_prof = st.sidebar.file_uploader("Raw data (pre-processed)", type=upload_types, key="prof_raw")
clust_prof = st.sidebar.file_uploader("Cluster results file", type=upload_types, key="prof_clust")

if raw_prof and clust_prof:
    # read raw profile file
    ext_raw = os.path.splitext(raw_prof.name.lower())[1]
    ext_clust = os.path.splitext(clust_prof.name.lower())[1]

    if ext_raw in (".xlsx", ".xls"):
        raw_df = pd.read_excel(raw_prof)
    else:
        raw_df = pd.read_csv(raw_prof)

    if ext_clust in (".xlsx", ".xls"):
        clust_df = pd.read_excel(clust_prof)
    else:
        clust_df = pd.read_csv(clust_prof)

    raw_df["unique_id"] = raw_df.index
    clust_df["unique_id"] = clust_df.index

    # pick the cluster column
    cluster_opts = [c for c in clust_df.columns if re.search(r"cluster", c, re.I)]
    if not cluster_opts:
        st.error("No column matching 'cluster' found in your results file.")
        st.stop()
    prof_col = st.sidebar.selectbox("Which column is your cluster ID?", cluster_opts)

    # merge & relabel clusters (1-based, strings)
    merged = (
        raw_df.merge(clust_df[["unique_id", prof_col]], on="unique_id")
        .rename(columns={prof_col: "cluster_label"})
        .drop(columns=["unique_id"])
    )
    merged["cluster_label"] = (merged["cluster_label"].astype(int) + 1).astype(str)

    # compute counts
    counts = (
        merged["cluster_label"]
        .value_counts()
        .sort_index(key=lambda idx: idx.astype(int))
        .rename_axis("cluster_label")
        .reset_index(name="count")
    )

    # allow renaming
    st.sidebar.subheader("Rename Clusters")
    name_map = {}
    for cl in counts["cluster_label"]:
        name_map[cl] = st.sidebar.text_input(
            f"Label for cluster {cl}", value=cl, key=f"rename_{cl}"
        )
    counts["cluster_name"] = counts["cluster_label"].map(name_map)

    # plot with larger x-axis title
    st.markdown("### Cluster Counts")
    fig = px.bar(
        counts,
        x="cluster_name",
        y="count",
        labels={"cluster_name": "Cluster", "count": "Count"},
    )
    fig.update_xaxes(title_font_size=18, tickfont_size=14)

    fig.update_xaxes(
        type="category",
        tickmode="array",
        tickvals=counts["cluster_name"].tolist(),
        ticktext=[
            # if it’s a “1.0” style string, turn it into “1”, otherwise leave it alone
            str(int(float(x))) if re.fullmatch(r"-?\d+(\.0+)?", x) else x
            for x in counts["cluster_name"]
        ],
        title_font_size=18,
        tickfont_size=14,
    )

    fig.update_yaxes(title_font_size=18, tickfont_size=12)
    st.plotly_chart(fig, use_container_width=True)

    # download counts
    make_download(
        counts[["cluster_name", "count"]].rename(columns={"cluster_name": "cluster_label"}),
        f"{base_name}_cluster_counts",
        f"download_cluster_counts_{download_format}",
    )

    # now the profile stats
    extra = st.sidebar.multiselect("Additional stats", ["median", "min", "max"], default=[])
    stats = ["mean"] + extra

    progress = st.progress(0, text="Calculating profiles…")
    try:

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

        profiles = _get_profiles(merged, "cluster_label", stats).set_index("cluster_label")
        progress.progress(100, text="Profiles ready!")
    except Exception as e:
        st.error(f"Cluster profiling failed: {e}")
    finally:
        progress.empty()

    # download & show profiles
    make_download(
        profiles.reset_index(), f"{base_name}_profiles", f"download_profiles_{download_format}"
    )
    st.markdown("### Cluster Profiles")
    st.dataframe(profiles.T, use_container_width=True)
