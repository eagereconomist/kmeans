import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tennis_racquet_analysis.config import FIGURES_DIR, DATA_DIR

sns.set_theme(
    style="ticks",
    font_scale=1.2,
    rc={"axes.spines.right": False, "axes.spines.top": False},
)


def histogram(
    input_file: str,
    dir_label: str,
    x_axis: str,
    num_bins: int,
    output_dir: Path = FIGURES_DIR,
) -> pd.DataFrame:
    """
    Load data/{dir_label}/{input_file}, plot a Seaborn histplot of `x_axis`,
    save to `output_dir`, return the DataFrame.
    """
    input_path = DATA_DIR / dir_label / input_file
    df = pd.read_csv(input_path)
    if x_axis not in df.columns:
        raise ValueError(f"Column '{x_axis} not found in {input_path}")
    palette = sns.cubehelix_palette(
        n_colors=8, start=3, rot=1, reverse=True, gamma=0.4, light=0.7, dark=0.1
    )
    plt.rc("axes", prop_cycle=plt.cycler("color", palette))
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x=x_axis, bins=num_bins)
    ax.set(
        xlabel=x_axis.capitalize(),
        ylabel="Frequency",
        title=f"{x_axis.capitalize()} Histogram from {dir_label.capitalize()} DataFrame",
    )
    stem = Path(input_file).stem
    output_path = output_dir / f"{stem}_{x_axis}_hist.png"
    fig.savefig(output_path)
    plt.close(fig)
    return df


def scatter_plot(
    input_file: str,
    dir_label: str,
    x_axis: str,
    y_axis: str,
    output_dir: Path = FIGURES_DIR,
) -> pd.DataFrame:
    """
    Load data/{dir_label}/{input_file}, plot `x_axis` vs. `y_axis` scatter,
    save to `output_dir`, return the DataFrame.
    """
    input_path = DATA_DIR / dir_label / input_file
    df = pd.read_csv(input_path)
    missing = [col for col in (x_axis, y_axis) if col not in df.columns]
    if missing:
        raise ValueError(f"Column(s) {missing} not found in {input_path}")
    palette = sns.cubehelix_palette(
        n_colors=8, start=3, rot=1, reverse=True, gamma=0.4, light=0.7, dark=0.1
    )
    plt.rc("axes", prop_cycle=plt.cycler("color", palette))
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
    ax.set(
        xlabel=x_axis.capitalize(),
        ylabel=y_axis.capitalize(),
        title=f"{x_axis.capitalize()} vs. {y_axis.capitalize()} Scatterplot from {dir_label.capitalize()} DataFrame",
    )
    stem = Path(input_file).stem
    output_path = output_dir / f"{stem}_{x_axis}_scatter.png"
    fig.savefig(output_path)
    plt.close(fig)
    return df


def box_plot(
    input_file: str,
    dir_label: str,
    y_axis: str,
    brand: str = None,
    output_dir: Path = FIGURES_DIR,
) -> pd.DataFrame:
    """
    Load data/{dir_label}/{input_file}, plot boxplot of `x_axis` and `y_axis`,
    save to `output_dir`, return the DataFrame.
    """
    input_path = DATA_DIR / dir_label / input_file
    df = pd.read_csv(input_path)
    if y_axis not in df.columns:
        raise ValueError(f"Column '{y_axis}' not found in {input_path}")
    df["Brand"] = df["Racquet"].apply(lambda s: re.findall(r"[A-Z][a-z]+", s)[0])
    brands = sorted(df["Brand"].unique())
    if brand:
        if brand not in brands:
            raise ValueError(f"Brand '{brand}' not found. Available brands are: {brands!r}")
        df = df[df["Brand"] == brand]
        x_col = "Racquet"
        categories = sorted(df[x_col].unique())
        stem_label = brand.lower()
    else:
        x_col = "Brand"
        categories = brands
        stem_label = "by_brand"
    palette = sns.cubehelix_palette(
        n_colors=len(brands),
        start=3,
        rot=1,
        reverse=True,
        gamma=0.4,
        light=0.7,
        dark=0.1,
    )
    sns.set_style("ticks")
    plt.rc("axes", prop_cycle=plt.cycler("color", palette))
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x=x_col,
        y=y_axis,
        order=categories,
        palette=palette,
        ax=ax,
    )
    sns.despine(ax=ax)
    ax.set(
        xlabel="Brand",
        ylabel=y_axis.capitalize(),
        title=(f"{y_axis.capitalize()} Box Plot by Racquet Brand"),
    )
    stem = Path(input_file).stem
    output_dir = output_dir / f"{stem}_{stem_label}_{y_axis}_boxplot.png"
    fig.savefig(output_dir)
    plt.close(fig)
    return df


def violin_plot(
    input_file: str,
    dir_label: str,
    y_axis: str,
    brand: str = None,
    inner: str = "box",
    output_dir: Path = FIGURES_DIR,
) -> pd.DataFrame:
    """
    Load data/{dir_label}/{input_file}, plot violinplot of `x_axis` and `y_axis`,
    save to `output_dir`, return the DataFrame.
    """
    input_path = DATA_DIR / dir_label / input_file
    df = pd.read_csv(input_path)
    if y_axis not in df.columns:
        raise ValueError(f"Column '{y_axis}' not found in {input_path}")
    df["Brand"] = df["Racquet"].apply(lambda s: re.findall(r"[A-Z][a-z]+", s)[0])
    brands = sorted(df["Brand"].unique())
    if brand:
        if brand not in brands:
            raise ValueError(f"Brand '{brand}' not found. Available brands are: {brands!r}")
        df = df[df["Brand"] == brand]
        x_col = "Racquet"
        categories = sorted(df[x_col].unique())
        stem_label = brand.lower()
    else:
        x_col = "Brand"
        categories = brands
        stem_label = "by_brand"
    palette = sns.cubehelix_palette(
        n_colors=len(brands),
        start=3,
        rot=1,
        reverse=True,
        gamma=0.4,
        light=0.7,
        dark=0.1,
    )
    sns.set_style("ticks")
    plt.rc("axes", prop_cycle=plt.cycler("color", palette))
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(
        data=df,
        x=x_col,
        y=y_axis,
        order=categories,
        palette=palette,
        inner=inner,
        ax=ax,
    )
    sns.despine(ax=ax)
    ax.set(
        xlabel="Brand",
        ylabel=y_axis.capitalize(),
        title=(f"{y_axis.capitalize()} Violin Plot by Racquet Brand"),
    )
    stem = Path(input_file).stem
    output_dir = output_dir / f"{stem}_{stem_label}_{y_axis}_violinplot.png"
    fig.savefig(output_dir)
    plt.close(fig)
    return df
