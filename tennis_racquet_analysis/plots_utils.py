import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tennis_racquet_analysis.config import FIGURES_DIR, DATA_DIR

PALETTE = sns.cubehelix_palette(8, start=2, rot=0.3)

plt.rc("axes", prop_cycle=plt.cycler("color", PALETTE))


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
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x=x_axis, bins=num_bins, ax=ax)
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
    x_axis: str,
    y_axis: str,
    output_dir: Path = FIGURES_DIR,
) -> pd.DataFrame:
    """
    Load data/{dir_label}/{input_file}, plot boxplot of `x_axis` vs. `y_axis`,
    save to `output_dir`, return the DataFrame.
    """
    input_path = DATA_DIR / dir_label / input_file
    df = pd.read_csv(input_path)
    missing = [col for col in (x_axis, y_axis) if col not in df.columns]
    if missing:
        raise ValueError(f"Column(s) {missing} not found in {input_path}")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x=x_axis, y=y_axis, ax=ax)
    ax.set(
        xlabel=x_axis.capitalize(),
        ylabel=y_axis.capitalize(),
        title=f"{x_axis.capitalize()} vs. {y_axis.capitalize()} Box Plot from {dir_label.capitalize()} DataFrame",
    )
    stem = Path(input_file).stem
    output_path = output_dir / f"{stem}_{x_axis}_boxplot.png"
    fig.savefig(output_path)
    plt.close(fig)
    return df
