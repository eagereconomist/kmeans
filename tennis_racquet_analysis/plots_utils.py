import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
import re


sns.set_theme(
    style="ticks",
    font_scale=1.2,
    rc={"axes.spines.right": False, "axes.spines.top": False},
)


def _init_fig(figsize=(10, 6)):
    """
    Create a fig + ax with shared cubehelix palette.
    """
    palette = sns.cubehelix_palette(
        n_colors=8, start=3, rot=1, reverse=True, light=0.7, dark=0.1, gamma=0.4
    )
    plt.rc("axes", prop_cycle=plt.cycler("color", palette))
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _save_fig(fig: plt.Figure, path: Path):
    """
    Ensure directory exists, save and close.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _set_axis_bounds(ax, vals: pd.Series, axis: str = "x"):
    lower, higher = 0, vals.max() + 1
    if axis == "x":
        ax.set_xlim(lower, higher)
    else:
        ax.set_ylim(lower, higher)


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
    ax.set_xlim(0, vals.max() + 1)
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
