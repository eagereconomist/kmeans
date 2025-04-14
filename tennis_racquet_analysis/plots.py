from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import matplotlib.pyplot as plt
from itertools import groupby

from tennis_racquet_analysis.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):


if __name__ == "__main__":
    app()
