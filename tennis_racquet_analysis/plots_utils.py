import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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
    """
    Set axis limits so that
        - lower = min(vals.min(), 0)
        - upper = vals.max() + 1
        axis: "x" or "y"
        pad: absolute padding to add to the high end
    """
    lower = min(vals.min(), 0)
    higher = vals.max() + 1
    if axis == "x":
        ax.set_xlim(lower, higher)
    else:
        ax.set_ylim(lower, higher)


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
        ax, fig = _init_fig()
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
