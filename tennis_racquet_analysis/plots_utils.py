import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Sequence
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
import statsmodels.api as sm
import re
import plotly.express as px

sns.set_theme(
    style="ticks",
    font_scale=1.2,
    rc={"axes.spines.right": False, "axes.spines.top": False},
)


def _init_fig(figsize=(20, 14)):
    """
    Create a fig + ax with shared cubehelix palette.
    """
    _apply_cubehelix_style()
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _apply_cubehelix_style():
    palette = sns.cubehelix_palette(
        n_colors=8, start=3, rot=1, reverse=True, light=0.7, dark=0.1, gamma=0.4
    )
    plt.rc("axes", prop_cycle=plt.cycler("color", palette))


def _set_axis_bounds(ax, vals: pd.Series, axis: str = "x"):
    lower, higher = 0, vals.max() + 1
    if axis == "x":
        ax.set_xlim(lower, higher)
    else:
        ax.set_ylim(lower, higher)


def _save_fig(fig: plt.Figure, path: Path):
    """
    Ensure directory exists, save and close.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def df_to_array(df: pd.DataFrame, columns: list[str] | None = None) -> np.ndarray:
    if columns:
        return df[columns].to_numpy()
    return df.select_dtypes(include="number").to_numpy()


def df_to_labels(
    df: pd.DataFrame,
    label_col: str,
) -> np.ndarray:
    return df[label_col].astype(str).to_numpy()


def compute_linkage(
    array: np.ndarray,
    method: str = "centroid",
    metric: str = "euclidean",
    optimal_ordering: bool = True,
) -> np.ndarray:
    dists = pdist(array, metric=metric)
    return sch.linkage(y=dists, method=method, metric=metric, optimal_ordering=optimal_ordering)


def dendrogram_plot(
    Z: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    orient: str = "right",
    save: bool = True,
    ax: plt.Axes | None = None,
) -> dict:
    if ax is None:
        fig, ax = _init_fig()
    else:
        fig = ax.figure
    result = (
        sch.dendrogram(
            Z,
            labels=labels,
            orientation=orient,
            ax=ax,
        ),
        ax.set(
            title=("Hierarchical Clustering Dendrogram (all leaves)"),
            xlabel=("Distance"),
            ylabel=("Racquet Model"),
        ),
    )
    if save:
        _save_fig(fig, output_path)
    return result


def histogram(
    df: pd.DataFrame,
    x_axis: str,
    num_bins: int,
    output_path: Path,
    save: bool = True,
    ax: plt.Axes = None,
) -> pd.DataFrame:
    if x_axis not in df.columns:
        raise ValueError(f"Column '{x_axis}' not in DataFrame.")
    if ax is None:
        fig, ax = _init_fig()
    else:
        fig = ax.figure
    sns.histplot(data=df, x=x_axis, bins=num_bins, ax=ax)
    vals = df[x_axis]
    _set_axis_bounds(ax, vals, axis="x")
    ax.set(
        xlabel=x_axis.capitalize(), ylabel="Frequency", title=f"Histogram of {x_axis.capitalize()}"
    )
    if save:
        _save_fig(fig, output_path)
    return df


def scatter_plot(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    output_path: Path,
    save: bool = True,
    ax: plt.Axes = None,
) -> pd.DataFrame:
    missing = [col for col in (x_axis, y_axis) if col not in df.columns]
    if missing:
        raise ValueError(f"Columns {missing} not in DataFrame.")
    if ax is None:
        fig, ax = _init_fig()
    else:
        fig = ax.figure
    sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
    x_vals, y_vals = df[x_axis], df[y_axis]
    _set_axis_bounds(ax, x_vals, axis="x")
    _set_axis_bounds(ax, y_vals, axis="y")
    ax.set(
        xlabel=x_axis.capitalize(),
        ylabel=y_axis.capitalize(),
        title=f"{x_axis.capitalize()} vs. {y_axis.capitalize()}",
    )
    if save:
        _save_fig(fig, output_path)
        return df


def box_plot(
    df: pd.DataFrame,
    y_axis: str,
    output_path: Path,
    brand: str = None,
    orient: str = "v",
    save: bool = True,
    ax: plt.Axes = None,
) -> pd.DataFrame:
    if y_axis not in df.columns:
        raise ValueError(f"Column '{y_axis}' not found in DataFrame.")
    df["Brand"] = df["Racquet"].apply(lambda s: re.findall(r"[A-Z][a-z]+", s)[0])
    all_brands = sorted(df["Brand"].unique())
    if brand:
        if brand not in all_brands:
            raise ValueError(f"Brand '{brand}' not one of the available brands: {all_brands!r}")
        df = df[df["Brand"] == brand]
        x_col = "Racquet"
        order = sorted(df[x_col].unique())
    else:
        x_col = "Brand"
        order = all_brands
    if ax is None:
        fig, ax = _init_fig()
    else:
        fig = ax.figure
    sns.boxplot(
        data=df,
        x=(x_col if orient == "v" else y_axis),
        y=(y_axis if orient == "v" else x_col),
        order=order,
        orient=orient,
        ax=ax,
    )
    vals = df[y_axis]
    if orient.lower().startswith("h"):
        _set_axis_bounds(ax, vals, axis="x")
        xlabel, ylabel = x_col, y_axis
    else:
        _set_axis_bounds(ax, vals, axis="y")
        xlabel, ylabel = (x_col, y_axis) if orient.lower().startswith("v") else (y_axis, x_col)
    ax.set(
        xlabel=xlabel.capitalize(),
        ylabel=ylabel.capitalize(),
        title=f"Box Plot of {y_axis.capitalize()} for {brand or 'All Brands'}",
    )
    if save:
        _save_fig(fig, output_path)
    return df


def violin_plot(
    df: pd.DataFrame,
    y_axis: str,
    output_path: Path,
    brand: str = None,
    orient: str = "v",
    inner: str = "box",
    save: bool = True,
    ax: plt.Axes = None,
) -> pd.DataFrame:
    if y_axis not in df.columns:
        raise ValueError(f"Column '{y_axis}' not found in DataFrame.")
    df["Brand"] = df["Racquet"].apply(lambda s: re.findall(r"[A-Z][a-z]+", s)[0])
    all_brands = sorted(df["Brand"].unique())
    if brand:
        if brand not in all_brands:
            raise ValueError(f"Brand '{brand}' not one of the available brands: {all_brands!r}")
        df = df[df["Brand"] == brand]
        x_col = "Racquet"
        order = sorted(df[x_col].unique())
    else:
        x_col = "Brand"
        order = all_brands
    if ax is None:
        fig, ax = _init_fig()
    else:
        fig = ax.figure
    sns.violinplot(
        data=df,
        x=(x_col if orient == "v" else y_axis),
        y=(y_axis if orient == "v" else x_col),
        order=order,
        orient=orient,
        inner=inner,
        ax=ax,
    )
    vals = df[y_axis]
    if orient.lower().startswith("h"):
        _set_axis_bounds(ax, vals, axis="x")
        xlabel, ylabel = x_col, y_axis
    else:
        _set_axis_bounds(ax, vals, axis="y")
        xlabel, ylabel = (x_col, y_axis) if orient.lower().startswith("v") else (y_axis, x_col)
    ax.set(
        xlabel=xlabel.capitalize(),
        ylabel=ylabel.capitalize(),
        title=f"Violin Plot of {y_axis.capitalize()} for {brand or 'All Brands'}",
    )
    if save:
        _save_fig(fig, output_path)
    return df


def correlation_matrix_heatmap(
    df: pd.DataFrame,
    output_path: Path,
    save: bool = True,
    ax: plt.Axes = None,
) -> pd.DataFrame:
    corr = df.corr(method="pearson")
    if ax is None:
        fig, ax = _init_fig()
    else:
        fig = ax.figure
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        center=0,
        cmap="crest",
        linewidths=0.5,
        vmin=-1,
        vmax=1,
        ax=ax,
    )
    ax.set(
        title=f"Correlation Matrix Heatmap {output_path.stem}",
        xlabel="Features",
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    if save:
        _save_fig(fig, output_path)
    return df


def qq_plot(
    df: pd.DataFrame, column: str, output_path: Path, save: bool = True, ax: plt.Axes | None = None
) -> plt.Axes:
    if column not in df.columns:
        raise ValueError(f"Column {column!r} not found")
    series = df[column]
    if ax is None:
        fig, ax = _init_fig()
    sm.qqplot(series, line="r", ax=ax)
    ax.set_title(f"Q-Q Plot: {column.capitalize()}")
    if save:
        _save_fig(fig, output_path)
    return ax


def inertia_plot(inertia_df: pd.DataFrame, output_path: Path, save: bool = True) -> plt.Axes:
    fig, ax = _init_fig()
    ax.plot(inertia_df["k"], inertia_df["inertia"], marker="o")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title(f"Elbow Plot for K-Means Inertia from {output_path} ")
    if save:
        _save_fig(fig, output_path)
    return fig


def silhouette_plot(silhouette_df: pd.DataFrame, output_path: Path, save: bool = True) -> plt.Axes:
    fig, ax = _init_fig()
    ax.plot(silhouette_df["n_clusters"], silhouette_df["silhouette_score"], marker="o")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title(f"Silhouette Score vs. Number of Clusters from {output_path}")
    if save:
        _save_fig(fig, output_path)
    return fig


def scree_plot(
    df: pd.DataFrame,
    output_path: Path,
    save: bool = True,
) -> plt.Axes:
    fig, ax = _init_fig()
    x = range(1, len(df["prop_var"]) + 1)
    y = df["prop_var"].values
    ax.plot(x, y, marker="o")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Prop. Variance Explained")
    ax.set_title(f"Scree Plot ({output_path.name})")
    if save:
        _save_fig(fig, output_path)
    return fig


def cumulative_prop_var_plot(
    df: pd.DataFrame,
    output_path: Path,
    save: bool = True,
) -> plt.Axes:
    fig, ax = _init_fig()
    x = range(1, len(df["cumulative_prop_var"]) + 1)
    y = df["cumulative_prop_var"].values
    ax.plot(x, y, marker="o")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Cumulative Prop. Variance Explained")
    if save:
        _save_fig(fig, output_path)
    return fig


def pca_biplot(
    df: pd.DataFrame,
    loadings: pd.DataFrame,
    pve: pd.Series,
    pc_x: int = 0,
    pc_y: int = 1,
    scale: float = 1.0,
    figsize: tuple[float, float] = (20, 14),
    hue: Optional[Sequence] = None,
    save: bool = True,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    feature_cols = loadings.columns.tolist()
    X = df[feature_cols].values
    scores = X.dot(loadings.values.T)

    var_x = pve.iloc[pc_x]
    var_y = pve.iloc[pc_y]
    x_label = f"PC{pc_x + 1} ({var_x:.1%})"
    y_label = f"PC{pc_y + 1} ({var_y:.1%})"

    fig, ax = _init_fig(figsize=figsize)

    if hue is None:
        ax.scatter(scores[:, pc_x], scores[:, pc_y], alpha=1)
    else:
        cat_hue = pd.Categorical(hue)
        codes = cat_hue.codes
        categories = cat_hue.categories

        cmap = plt.get_cmap("tab10")

        ax.scatter(scores[:, pc_x], scores[:, pc_y], c=codes, cmap=cmap, alpha=0.7)

        handles = [
            Line2D([], [], marker="o", color=cmap(i), linestyle="", markersize=6)
            for i in range(len(categories))
        ]
        labels = [str(cat) for cat in categories]
        ax.legend(handles, labels, title="cluster", loc="best")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("PCA Biplot", pad=40, fontdict={"fontsize": 30})

    for k, feature in enumerate(feature_cols):
        x_arr = loadings.iat[pc_x, k] * scale
        y_arr = loadings.iat[pc_y, k] * scale
        ax.arrow(
            0,
            0,
            x_arr,
            y_arr,
            head_width=0.02 * scale,
            head_length=0.02 * scale,
            length_includes_head=True,
            color="black",
        )
        ax.text(x_arr * 1.1, y_arr * 1.1, feature, fontsize=10)

    if save and output_path is not None:
        _save_fig(fig, output_path)

    return fig


def cluster_scatter(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    output_path: Path,
    label_column: str = "cluster",
    save: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        fig, ax = _init_fig()
    else:
        fig = ax.figure
    sns.scatterplot(
        data=df,
        x=x_axis,
        y=y_axis,
        style=label_column,
        marker=True,
        hue=label_column,
        palette="dark",
        legend="full",
    )
    ax.set_title(f"{x_axis.capitalize()} vs. {y_axis.capitalize()} by {label_column}")
    if save:
        _save_fig(fig, output_path)
    return ax


def cluster_scatter_3d(
    df: pd.DataFrame,
    features: list[str],
    label_column: str,
    output_path: Path,
    save: bool = True,
) -> px.scatter_3d:
    if len(features) < 3:
        raise ValueError("Need at least 3 features for a 3D plot.")
    x, y, z = features[:3]
    cluster_order = sorted(df[label_column].unique(), key=lambda x: int(x))
    fig = px.scatter_3d(
        df,
        x=x,
        y=y,
        z=z,
        color=label_column,
        category_orders={label_column: cluster_order},
        title=f"3D Cluster Scatter (k={label_column.split('_')[-1]})",
        width=1400,
        height=1000,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(output_path))
    return fig


def plot_batch_clusters(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    cluster_columns: list[str],
    output_path: Path,
    save: bool = True,
    columns_per_row: int = 3,
    figsize_per_plot: tuple[int, int] = (12, 12),
) -> plt.Figure:
    n = len(cluster_columns)
    columns = columns_per_row or n
    rows = (n + columns - 1) // columns
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * figsize_per_plot[0], rows * figsize_per_plot[1]),
        squeeze=False,
    )
    for ax, column in zip(axes.flat, cluster_columns):
        sns.scatterplot(
            data=df,
            x=x_axis,
            y=y_axis,
            hue=column,
            style=column,
            markers=True,
            s=100,
            alpha=1,
            palette="dark",
            ax=ax,
            edgecolor="grey",
            legend="full",
        )
        ax.set_title(f"{column} ")
        ax.set_xlabel(x_axis.capitalize())
        ax.set_ylabel(y_axis.capitalize())
    for ax in axes.flat[n:]:
        fig.delaxes(ax)
    fig.tight_layout()
    if save:
        _save_fig(fig, output_path)
    return fig
