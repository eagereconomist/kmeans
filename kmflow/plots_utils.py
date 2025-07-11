import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Sequence
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import re
import plotly.express as px
import plotly.graph_objects as go


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


def _prepare_category(
    df: pd.DataFrame,
    category_col: str,
    patterns: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Create a 'Category' column from df[category_col], then optionally
    fileter and sort its unique values via regex patterns. Returns a tuple:
    (df_with_Category, ordered list of categories).
    """
    if category_col not in df.columns:
        raise ValueError(f"Column '{category_col}' not found in DataFrame.")
    df_plot = df.copy()
    df_plot["Category"] = df_plot[category_col].astype(str)
    categories = sorted(df_plot["Category"].unique())
    if patterns:
        filtered = [cat for cat in categories if any(re.search(pat, cat) for pat in patterns)]
        if not filtered:
            raise ValueError(
                f"No categories match patterns {patterns!r}\n"
                "Hint: when specifying multiple patterns, separate them with\n"
                "a comma and space, (e.g. -p 'Price, Weight')."
            )
        df_plot = df_plot[df_plot["Category"].isin(filtered)]
        categories = filtered

    return df_plot, categories


def _ensure_unique_path(path: Path) -> Path:
    """
    If `path` already exists, append _1, _2, ... before the suffix
    until we find a filename that doesn't exist.
    """
    if not path.exists():
        return path
    base, ext = path.stem, path.suffix
    i = 1
    while True:
        candidate = path.parent / f"{base}_{i}{ext}"
        if not candidate.exists():
            return candidate
        i += 1


def _save_fig(fig: plt.Figure, path: Path):
    """
    Ensure directory exists, save and close.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def bar_plot(
    df: pd.DataFrame,
    category_col: str,
    numeric_col: str,
    output_path: Path,
    orient: str = "v",
    save: bool = True,
    ax: plt.Axes = None,
) -> pd.DataFrame:
    for col in (category_col, numeric_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
    df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce")
    if df[numeric_col].isna().any():
        raise ValueError(f"Column '{numeric_col}' must be category.")
    if ax is None:
        fig, ax = _init_fig()
    else:
        fig = ax.figure
    if orient.lower().startswith("h"):
        x, y = numeric_col, category_col
    else:
        x, y = category_col, numeric_col
    sns.barplot(data=df, x=x, y=y, orient=orient, ax=ax)
    _set_axis_bounds(ax, df[numeric_col], axis=("x" if orient.lower().startswith("h") else "y"))
    ax.set(
        xlabel=x.replace("_", " ").capitalize(),
        ylabel=y.replace("_", " ").capitalize(),
        title=f"Bar Plot of {numeric_col.replace('_', ' ').capitalize()} by {category_col.replace('_', ' ').capitalize()}",
    )
    if save:
        _save_fig(fig, output_path)
    return df


def histogram(
    df: pd.DataFrame,
    num_bins: int,
    x_axis: str,
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
    scale: float = 1.0,
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

    if scale != 1.0:
        x0, x1 = ax.get_xlim()
        x_mid = (x0 + x1) / 2
        half_width = (x1 - x0) / 2 * scale
        ax.set_xlim(x_mid - half_width, x_mid + half_width)

        y0, y1 = ax.get_ylim()
        y_mid = (y0 + y1) / 2
        half_height = (y1 - y0) / 2 * scale
        ax.set_ylim(y_mid - half_height, y_mid + half_height)
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
    numeric_col: str,
    output_path: Path,
    category_col: Optional[str] = None,
    patterns: Optional[Sequence[str]] = None,
    orient: str = "v",
    save: bool = True,
    ax: Optional[plt.Axes] = None,
) -> pd.DataFrame:
    """
    Draw a box plot of numeric_col.
    - If category_col is None: one global box.
    - Else: one box per category extracted from df[category_col];
      if patterns is given, only matching categories are shown.
    """
    if numeric_col not in df.columns:
        raise ValueError(f"Column '{numeric_col}' not found in DataFrame.")
    if category_col:
        df_plot, order = _prepare_category(df, category_col, patterns)
        x_col = "Category"
    else:
        df_plot = df.copy()
        x_col = None
        order = None
    if ax is None:
        fig, ax = _init_fig()
    else:
        fig = ax.figure
    if x_col is None:
        sns.boxplot(data=df_plot, y=numeric_col, orient=orient, ax=ax)
        _set_axis_bounds(ax, df_plot[numeric_col], axis="y")
    else:
        sns.boxplot(
            data=df_plot,
            x=(x_col if orient.lower().startswith("v") else numeric_col),
            y=(numeric_col if orient.lower().startswith("v") else x_col),
            order=order,
            orient=orient,
            ax=ax,
        )
        vals = df_plot[numeric_col]
        axis = "y" if orient.lower().startswith("v") else "x"
        _set_axis_bounds(ax, vals, axis=axis)
    if category_col:
        title = f"Box Plot of {numeric_col.capitalize()} by {category_col}"
    else:
        title = f"Box Plot of {numeric_col.capitalize()}"
    ax.set(
        xlabel=None if x_col is None and orient.lower().startswith("h") else (x_col or ""),
        ylabel=None
        if x_col is None and orient.lower().startswith("v")
        else numeric_col.capitalize(),
        title=title,
    )
    if save:
        _save_fig(fig, output_path)
    return df_plot


def violin_plot(
    df: pd.DataFrame,
    numeric_col: str,
    output_path: Path,
    category_col: Optional[str] = None,
    patterns: Optional[Sequence[str]] = None,
    orient: str = "v",
    inner: str = "box",
    save: bool = True,
    ax: Optional[plt.Axes] = None,
) -> pd.DataFrame:
    """
    Draw a violin plot of numeric_col.
    - If category_col is None: one global violin.
    - Else: one violin per category extracted from df[category_col];
      if patterns is given, only matching categories are shown.
    """
    if numeric_col not in df.columns:
        raise ValueError(f"Column '{numeric_col}' not found in DataFrame.")
    if category_col:
        df_plot, order = _prepare_category(df, category_col, patterns)
        x_col = "Category"
    else:
        df_plot = df.copy()
        x_col = None
        order = None
    if ax is None:
        fig, ax = _init_fig()
    else:
        fig = ax.figure
    if x_col is None:
        if orient.lower().startswith("h"):
            sns.violinplot(data=df_plot, x=numeric_col, orient=orient, inner=inner, ax=ax)
            _set_axis_bounds(ax, df_plot[numeric_col], axis="x")
        else:
            sns.violinplot(data=df_plot, y=numeric_col, orient=orient, inner=inner, ax=ax)
            _set_axis_bounds(ax, df_plot[numeric_col], axis="y")
    else:
        sns.violinplot(
            data=df_plot,
            x=(x_col if orient.lower().startswith("v") else numeric_col),
            y=(numeric_col if orient.lower().startswith("v") else x_col),
            order=order,
            orient=orient,
            inner=inner,
            ax=ax,
        )
        vals = df_plot[numeric_col]
        axis = "y" if orient.lower().startswith("v") else "x"
        _set_axis_bounds(ax, vals, axis=axis)
    if category_col:
        title = f"Violin Plot of {numeric_col.capitalize()} by {category_col}"
    else:
        title = f"Violin Plot of {numeric_col.capitalize()}"
    ax.set(
        xlabel=None if x_col is None and orient.lower().startswith("h") else (x_col or ""),
        ylabel=None
        if x_col is None and orient.lower().startswith("v")
        else numeric_col.capitalize(),
        title=title,
    )
    if save:
        _save_fig(fig, output_path)
    return df_plot


def corr_heatmap(
    df: pd.DataFrame,
    output_path: Path,
    save: bool = True,
    ax: plt.Axes = None,
) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        raise ValueError("No numeric columns found for correlation heatmap.")
    corr = numeric_df.corr(method="pearson")
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
        title="Correlation Matrix Heatmap",
        xlabel="Features",
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    if save:
        _save_fig(fig, output_path)
    return df


def qq_plot(
    df: pd.DataFrame,
    numeric_col: str,
    output_path: Path,
    save: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if numeric_col not in df.columns:
        raise ValueError(f"numeric_col {numeric_col!r} not found")
    series = df[numeric_col]
    if ax is None:
        fig, ax = _init_fig()
    sm.qqplot(series, line="r", ax=ax)
    ax.set_title(f"Q-Q Plot: {numeric_col.capitalize()}")
    if save:
        _save_fig(fig, output_path)
    return ax


def inertia_plot(inertia_df: pd.DataFrame, output_path: Path, save: bool = True) -> plt.Axes:
    fig, ax = _init_fig()
    ax.plot(inertia_df["k"], inertia_df["inertia"], marker="o")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Inertia Plot")
    ax.set_xticks(inertia_df["k"].tolist())
    if save:
        _save_fig(fig, output_path)
    return fig


def silhouette_plot(silhouette_df: pd.DataFrame, output_path: Path, save: bool = True) -> plt.Axes:
    fig, ax = _init_fig()
    ax.plot(silhouette_df["n_clusters"], silhouette_df["silhouette_score"], marker="o")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score vs. k")
    ax.set_xticks(silhouette_df["n_clusters"].tolist())
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
    ax.set_title("Scree Plot")
    if save:
        _save_fig(fig, output_path)
    return fig


def cumulative_var_plot(
    df: pd.DataFrame,
    output_path: Path,
    save: bool = True,
) -> plt.Axes:
    fig, ax = _init_fig()
    x = range(1, len(df["cumulative_prop_var"]) + 1)
    y = df["cumulative_prop_var"].values
    ax.plot(x, y, marker="o")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Cumulative Proportion of Variance Explained")
    if save:
        _save_fig(fig, output_path)
    return fig


def cluster_scatter(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    output_path: Path,
    scale: float = 1.0,
    cluster_col: str = "cluster",
    save: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    df_plot = df.copy()
    df_plot[cluster_col] = df_plot[cluster_col].astype(int)
    df_plot[cluster_col] = (df_plot[cluster_col] + 1).astype(str)
    categories = sorted(df_plot[cluster_col].unique(), key=lambda v: int(v))
    if ax is None:
        fig, ax = _init_fig()
    else:
        fig = ax.figure
    sns.scatterplot(
        data=df_plot,
        x=x_axis,
        y=y_axis,
        style=cluster_col,
        marker=True,
        hue=cluster_col,
        hue_order=categories,
        style_order=categories,
        palette="dark",
        legend="full",
    )

    _set_axis_bounds(ax, df[x_axis], axis="x")
    _set_axis_bounds(ax, df[y_axis], axis="y")

    if scale != 1.0:
        x0, x1 = ax.get_xlim()
        x_mid = (x0 + x1) / 2
        half_w = (x1 - x0) / 2 * scale
        ax.set_xlim(x_mid - half_w, x_mid + half_w)

        y0, y1 = ax.get_ylim()
        y_mid = (y0 + y1) / 2
        half_h = (y1 - y0) / 2 * scale
        ax.set_ylim(y_mid - half_h, y_mid + half_h)

    ax.set_title(f"{x_axis.capitalize()} vs. {y_axis.capitalize()} by {cluster_col}")
    if save:
        _save_fig(fig, output_path)
    return ax


def cluster_scatter_3d(
    df: pd.DataFrame,
    numeric_cols: list[str],
    cluster_col: str,
    output_path: Path,
    scale: float = 1.0,
    save: bool = True,
) -> px.scatter_3d:
    if len(numeric_cols) != 3:
        raise ValueError(
            "Need exactly three numeric features for 3D plotting; e.g. 'weight' 'height' 'width'"
        )
    missing = [column for column in numeric_cols if column not in df.columns]
    if missing:
        raise KeyError(
            f"Column(s) {missing!r} not found in the DataFrame. "
            "Please choose three valid numeric columns, "
            "for example: 'beamwidth' 'headsize' 'length'."
        )
    df_plot = df.copy()
    df_plot[cluster_col] = df_plot[cluster_col].astype(int) + 1
    df_plot[cluster_col] = df_plot[cluster_col].astype(str)
    if len(numeric_cols) != 3:
        raise ValueError("Need at least 3 numeric_cols for a 3D plot.")
    df_scaled = df_plot.copy()
    for _, feat in enumerate(numeric_cols):
        df_scaled[feat] *= scale

    df_scaled[cluster_col] = df_scaled[cluster_col].astype(str)
    try:
        order = sorted(df_scaled[cluster_col].unique(), key=int)
    except ValueError:
        order = list(df_scaled[cluster_col].unique())
    fig = px.scatter_3d(
        df_scaled,
        x=numeric_cols[0],
        y=numeric_cols[1],
        z=numeric_cols[2],
        color=cluster_col,
        category_orders={cluster_col: order},
        title=f"3D Cluster Scatter (k={cluster_col.split('_')[-1]})",
        width=1000,
        height=1000,
    )
    if save:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(output_path))
    return fig


def plot_batch_clusters(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    cluster_cols: list[str],
    output_path: Path,
    save: bool = True,
    scale: float = 1.0,
    columns_per_row: int = 3,
    figsize_per_plot: tuple[int, int] = (12, 12),
) -> plt.Figure:
    """
    Create a grid of 2D cluster-colored scatters for each column in cluster_cols,
    ensuring each subplot uses its own cluster column.
    """
    n = len(cluster_cols)
    cols = columns_per_row or n
    rows = (n + cols - 1) // cols

    df_plot = df.copy()
    for col in cluster_cols:
        df_plot[col] = df_plot[col].astype(int) + 1
        df_plot[col] = df_plot[col].astype(str)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * figsize_per_plot[0], rows * figsize_per_plot[1]),
        squeeze=False,
    )

    for ax, column in zip(axes.flat, cluster_cols):
        categories = sorted(df_plot[column].unique(), key=lambda v: int(v))

        sns.scatterplot(
            data=df_plot,
            x=x_axis,
            y=y_axis,
            hue=column,
            style=column,
            palette="dark",
            s=100,
            alpha=1,
            edgecolor="grey",
            hue_order=categories,
            style_order=categories,
            ax=ax,
            legend="full",
        )
        ax.set_xlabel(x_axis.capitalize())
        ax.set_ylabel(y_axis.capitalize())
        ax.set_title(f"{column}")

        _set_axis_bounds(ax, df_plot[x_axis], axis="x")
        _set_axis_bounds(ax, df_plot[y_axis], axis="y")
        if scale != 1.0:
            x0, x1 = ax.get_xlim()
            xm = (x0 + x1) / 2
            width = (x1 - x0) / 2 * scale
            ax.set_xlim(xm - width, xm + width)

            y0, y1 = ax.get_ylim()
            ym = (y0 + y1) / 2
            height = (y1 - y0) / 2 * scale
            ax.set_ylim(ym - height, ym + height)

    for ax in axes.flat[n:]:
        fig.delaxes(ax)

    fig.tight_layout()
    if save:
        _save_fig(fig, output_path)

    return fig


def biplot(
    df: pd.DataFrame,
    loadings: pd.DataFrame,
    pve: pd.Series,
    compute_scores: bool = True,
    pc_x: int = 0,
    pc_y: int = 1,
    scale: float = 1.0,
    figsize: tuple[float, float] = (20, 14),
    hue: Optional[Sequence] = None,
    save: bool = True,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    feature_cols = loadings.columns.tolist()
    if compute_scores:
        X = df[feature_cols].values
        scores = X.dot(loadings.values.T)
    else:
        scores = df.values
    var_x, var_y = pve.iloc[pc_x], pve.iloc[pc_y]
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
        norm = Normalize(vmin=0, vmax=len(categories) - 1)

        ax.scatter(scores[:, pc_x], scores[:, pc_y], c=codes, cmap=cmap, norm=norm, alpha=1)

        handles = [
            Line2D([], [], marker="o", color=cmap(norm(i)), linestyle="", markersize=6)
            for i in range(len(categories))
        ]
        labels = [str(cat) for cat in categories]
        ax.legend(handles, labels, title="Cluster", loc="best")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"Biplot (k={hue.name.split('_')[-1]})", pad=40, fontdict={"fontsize": 30})

    if compute_scores:
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


def biplot_3d(
    df: pd.DataFrame,
    loadings: pd.DataFrame,
    pve: pd.Series,
    output_path: Path,
    compute_scores: bool = True,
    pc_x: int = 0,
    pc_y: int = 1,
    pc_z: int = 2,
    scale: float = 1.0,
    hue: pd.Series | None = None,
    save: bool = True,
) -> go.Figure:
    feature_cols = loadings.columns.tolist()
    if len(feature_cols) != 3:
        raise ValueError("Need exactly three features for a 3D plot.")
    if compute_scores:
        X = df[feature_cols].values
        scores = X.dot(loadings.values.T)
    else:
        scores = df.values
    x_vals, y_vals, z_vals = scores[:, pc_x], scores[:, pc_y], scores[:, pc_z]
    x_label = f"PC{pc_x + 1} ({pve.iloc[pc_x]:.1%})"
    y_label = f"PC{pc_y + 1} ({pve.iloc[pc_y]:.1%})"
    z_label = f"PC{pc_z + 1} ({pve.iloc[pc_z]:.1%})"
    plotly_df = pd.DataFrame(
        {
            x_label: x_vals,
            y_label: y_vals,
            z_label: z_vals,
        }
    )
    if hue is not None:
        hue_str = hue.astype(str).rename("cluster")
        plotly_df["cluster"] = hue_str
        try:
            order = sorted(hue_str.unique(), key=int)
        except ValueError:
            order = list(hue_str.unique())
    else:
        order = None
    fig = px.scatter_3d(
        plotly_df,
        x=x_label,
        y=y_label,
        z=z_label,
        color="cluster" if hue is not None else None,
        category_orders={"cluster": order} if order is not None else None,
        color_discrete_sequence=px.colors.qualitative.T10,
        labels={x_label: x_label, y_label: y_label, z_label: z_label},
        title=f"3D Biplot (k={hue.name.split('_')[-1] if hue is not None else ''})",
        width=1000,
        height=1000,
    )
    if compute_scores:
        head_length = scale * 0.04
        for i, feature in enumerate(feature_cols):
            xi = loadings.iat[pc_x, i] * scale
            yi = loadings.iat[pc_y, i] * scale
            zi = loadings.iat[pc_z, i] * scale
            fig.add_trace(
                go.Scatter3d(
                    x=[0, xi],
                    y=[0, yi],
                    z=[0, zi],
                    mode="lines",
                    line=dict(color="black", width=4),
                    showlegend=False,
                )
            )
            vec = np.array([xi, yi, zi])
            length = np.linalg.norm(vec)
            if length > 0:
                direction = vec / length
                ux, uy, uz = direction * head_length
            else:
                ux = uy = uz = 0
            fig.add_trace(
                go.Cone(
                    x=[xi],
                    y=[yi],
                    z=[zi],
                    u=[ux],
                    v=[uy],
                    w=[uz],
                    anchor="tip",
                    sizemode="absolute",
                    sizeref=head_length,
                    showscale=False,
                    colorscale=[[0, "black"], [1, "black"]],
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[xi],
                    y=[yi],
                    z=[zi],
                    mode="text",
                    text=[feature],
                    textposition="top center",
                    showlegend=False,
                )
            )
        fig.update_layout(legend=dict(title="Cluster", traceorder="normal"))
    if save and output_path != Path("-"):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(output_path))
    return fig
