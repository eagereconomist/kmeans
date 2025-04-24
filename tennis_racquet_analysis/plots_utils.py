import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tennis_racquet_analysis.config import DATA_DIR, FIGURES_DIR
from tennis_racquet_analysis.preprocessing_utils import load_data

sns.set_theme(
    style="ticks",
    font_scale=1.2,
    rc={"axes.spines.right": False, "axes.spines.top": False},
)


def _init_plot(n_colors: int):
    palette = sns.cubehelix_palette(
        n_colors=n_colors, start=3, rot=1, reverse=True, light=0.7, dark=0.1, gamma=0.4
    )
    plt.rc("axes", prop_cycle=plt.cycler("color", palette))
    return plt.subplots(figsize=(10, 6))


def _save_fig(fig: plt.Figure, stem: str, suffix: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_output = output_dir / f"{stem}_{suffix}.png"
    fig.savefig(figure_output)
    plt.close(fig)
    return figure_output


def histogram(
    input_file: str,
    dir_label: str,
    x_axis: str,
    num_bins: int,
    output_dir: Path = FIGURES_DIR,
) -> pd.DataFrame:
    input_path = DATA_DIR / dir_label / input_file
    df = load_data(input_path)
    if x_axis not in df.columns:
        raise ValueError(f"Column '{x_axis} not found in {input_path!r}")
    fig, ax = _init_plot(n_colors=8)
    sns.histplot(data=df, x=x_axis, bins=num_bins, ax=ax)
    ax.set(
        xlabel=x_axis.capitalize(),
        ylabel="Frequency",
        title=f"Histogram of {x_axis.capitalize()} from {dir_label.capitalize()} DataFrame",
    )
    stem = Path(input_file).stem
    _save_fig(fig, stem, f"{x_axis}_hist", output_dir)
    return df


def scatter_plot(
    input_file: str,
    dir_label: str,
    x_axis: str,
    y_axis: str,
    output_dir: Path = FIGURES_DIR,
) -> pd.DataFrame:
    input_path = DATA_DIR / dir_label / input_file
    df = load_data(input_path)
    for col in (x_axis, y_axis):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {input_path!r}")
    fig, ax = _init_plot(n_colors=8)
    sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
    ax.set(
        xlabel=x_axis.capitalize(),
        ylabel=y_axis.capitalize(),
        title=f"{x_axis.capitalize()} vs. {y_axis.capitalize()} Scatterplot from {dir_label.capitalize()} DataFrame",
    )
    stem = Path(input_file).stem
    _save_fig(fig, stem, f"{x_axis}_scatter", output_dir)
    return df


def box_plot(
    input_file: str,
    dir_label: str,
    y_axis: str,
    brand: str = None,
    orient: str = "v",
    output_dir: Path = FIGURES_DIR,
) -> pd.DataFrame:
    input_path = DATA_DIR / dir_label / input_file
    df = load_data(input_path)
    if y_axis not in df.columns:
        raise ValueError(f"Column '{y_axis}' not found in {input_path!r}")
    df["Brand"] = df["Racquet"].apply(lambda s: re.findall(r"[A-Z][a-z]+", s)[0])
    all_brands = sorted(df["Brand"].unique())
    if brand:
        if brand not in all_brands:
            raise ValueError(f"Brand '{brand}' not one of the available brands: {all_brands!r}")
        df = df[df["Brand"] == brand]
        x_col = "Racquet"
        order = sorted(df[x_col].unique())
        stem_label = brand.lower()
    else:
        x_col = "Brand"
        order = all_brands
        stem_label = "by_brand"
        fig, ax = _init_plot(n_colors=len(all_brands))
        sns.boxplot(
            data=df,
            x=(x_col if orient == "v" else y_axis),
            y=(y_axis if orient == "v" else x_col),
            order=order,
            orient=orient,
            ax=ax,
        )
        sns.despine(ax=ax)
        xlabel, ylabel = (x_col, y_axis) if orient == "v" else (y_axis, x_col)
        ax.set(
            xlabel=xlabel.capitalize(),
            ylabel=ylabel.capitalize(),
            title=f"Box Plot of {y_axis.capitalize()} for {brand or 'All Brands'}",
        )
        stem = Path(input_file).stem
        _save_fig(fig, stem, f"{stem_label}_{y_axis}_boxplot", output_dir)
        return df


def violin_plot(
    input_file: str,
    dir_label: str,
    y_axis: str,
    brand: str = None,
    orient: str = "v",
    inner: str = "box",
    output_dir: Path = FIGURES_DIR,
) -> pd.DataFrame:
    input_path = DATA_DIR / dir_label / input_file
    df = load_data(input_path)
    if y_axis not in df.columns:
        raise ValueError(f"Column '{y_axis}' not found in {input_path!r}")
    df["Brand"] = df["Racquet"].apply(lambda s: re.findall(r"[A-Z][a-z]+", s)[0])
    all_brands = sorted(df["Brand"].unique())
    if brand:
        if brand not in all_brands:
            raise ValueError(f"Brand '{brand}' not one of the available brands: {all_brands!r}")
        df = df[df["Brand"] == brand]
        x_col = "Racquet"
        order = sorted(df[x_col].unique())
        stem_label = brand.lower()
    else:
        x_col = "Brand"
        categories = all_brands
        stem_label = "by_brand"
        fig, ax = _init_plot(n_colors=len(all_brands))
        sns.violinplot(
            data=df,
            x=(x_col if orient == "v" else y_axis),
            y=(y_axis if orient == "v" else x_col),
            order=order,
            orient=orient,
            inner=inner,
            ax=ax,
        )
        sns.despine(ax=ax)
        xlabel, ylabel = (x_col, y_axis) if orient == "v" else (y_axis, x_col)
        ax.set(
            xlabel=xlabel.capitalize(),
            ylabel=ylabel.capitalize(),
            title=f"Violin Plot of {y_axis.capitalize()} for {brand or 'All Brands'}",
        )
        stem = Path(input_file).stem
        _save_fig(fig, stem, f"{stem_label}_{y_axis}_violin", output_dir)
        return df
