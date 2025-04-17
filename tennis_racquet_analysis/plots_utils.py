import pandas as pd
from tennis_racquet_analysis.config import (
    FIGURES_DIR,
    DATA_DIR,
    # INTERIM_DATA_DIR,
    # PROCESSED_DATA_DIR,
)  # noqa: F401
import matplotlib.pyplot as plt


def histogram(dir_label: str, file_label: str, output_path: str, x_axis: str, num_bins: int):
    file_path = DATA_DIR / dir_label / f"tennis_racquets_{file_label}.csv"
    output_path = FIGURES_DIR / f"{x_axis}_{file_label}_hist.png"
    df = pd.read_csv(file_path)
    if x_axis not in df.columns:
        raise ValueError(f"Column '{x_axis}' not found in the file: {file_path}")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df[x_axis], bins=num_bins, color="blue", edgecolor="black")
    ax.set_xlabel(x_axis.capitalize())
    ax.set_ylabel("Frequency")
    ax.set_title(f"Tennis Racquet {x_axis.capitalize()} from {file_label.capitalize()} Data")
    fig.savefig(output_path)
    return df


def scatter_plot(dir_label: str, file_label: str, output_path: str, x_axis: str, y_axis: str):
    file_path = DATA_DIR / dir_label / f"tennis_racquets_{file_label}.csv"
    output_path = FIGURES_DIR / f"{x_axis}_{y_axis}_{file_label}_scatter.png"
    df = pd.read_csv(file_path)
    missing = [col for col in (x_axis, y_axis) if col not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found in {file_path}: {', '.join(missing)}")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df[x_axis], df[y_axis], color="blue", edgecolors="blue")
    ax.set_xlabel(x_axis.capitalize())
    ax.set_ylabel(y_axis.capitalize())
    ax.set_title(
        f"Tennis Racquet {x_axis.capitalize()} and {y_axis.capitalize()} from {file_label.capitalize()} Data"
    )
    fig.savefig(output_path)
    return df
