import pandas as pd
from tennis_racquet_analysis.config import FIGURES_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR  # noqa: F401
import matplotlib.pyplot as plt


def histogram(dir_label: str, file_label: str, output_path: str, x_axis: str, num_bins: int = 30):
    file_path = INTERIM_DATA_DIR / f"tennis_racquets_{file_label}.csv"
    full_output_path = FIGURES_DIR / output_path
    df = pd.read_csv(file_path)
    if x_axis not in df.columns:
        raise ValueError(f"Column '{x_axis}' not found in the file: {file_path}")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df[x_axis], bins=num_bins, color="blue", edgecolor="black")
    ax.set_xlabel(x_axis.capitalize())
    ax.set_ylabel("Frequency")
    ax.set_title(f"Tennis Racquet {x_axis.capitalize()} from {file_label.capitalize()} Data")
    fig.savefig(full_output_path)
