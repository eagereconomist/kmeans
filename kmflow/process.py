from pathlib import Path

import typer

from kmflow.process_utils import (
    _run_scaler,
    apply_normalizer,
    apply_standardization,
    apply_minmax,
    apply_log1p,
    apply_yeo_johnson,
)
from kmflow.config import PROCESSED_DATA_DIR

app = typer.Typer(help="Apply individual scaler to a preprocessed CSV.")


@app.command("norm")
def normalize(
    input_file: Path = typer.Argument(..., help="CSV to read (use '-' to read from stdin)."),
    output_file: Path = typer.Option(
        PROCESSED_DATA_DIR,
        "--output-file",
        "-o",
        help="Where to save normalized CSV; use '-' for stdout. Defaults to processed directory.",
    ),
):
    """L2-normalize all numeric columns."""
    _run_scaler(apply_normalizer, input_file, output_file, "norm")


@app.command("std")
def standardize(
    input_file: Path = typer.Argument(
        ...,
        help="CSV to read (use '-' to read from stdin).",
    ),
    output_file: Path = typer.Option(
        PROCESSED_DATA_DIR,
        "--output-file",
        "-o",
        help="Where to save standardized CSV; use '-' for stdout. Defaults to processed directory.",
    ),
):
    """
    Z-score standardize all numeric columns.
    """
    _run_scaler(apply_standardization, input_file, output_file, "standardized")


@app.command("minmax")
def minmax(
    input_file: Path = typer.Argument(
        ...,
        help="CSV to read (use '-' to read from stdin).",
    ),
    output_file: Path = typer.Option(
        PROCESSED_DATA_DIR,
        "--output-file",
        "-o",
        help="Where to save min-max scaled CSV; use '-' for stdout. Defaults to processed directory.",
    ),
):
    """
    Scale all numeric columns to [0,1].
    """
    _run_scaler(apply_minmax, input_file, output_file, "minmax")


@app.command("log1p")
def log_scale(
    input_file: Path = typer.Argument(
        ...,
        help="CSV to read (use '-' to read from stdin).",
    ),
    output_file: Path = typer.Option(
        PROCESSED_DATA_DIR,
        "--output-file",
        "-o",
        help="Where to save log1p-transformed CSV; use '-' for stdout. Defaults to processed directory.",
    ),
):
    """
    Apply log(1 + x) transform to all numeric columns.
    """
    _run_scaler(apply_log1p, input_file, output_file, "log_scale")


@app.command("yj")
def yeo_johnson_scale(
    input_file: Path = typer.Argument(
        ...,
        help="CSV to read (use '-' to read from stdin).",
    ),
    output_file: Path = typer.Option(
        PROCESSED_DATA_DIR,
        "--output-file",
        "-o",
        help="Where to save Yeo-Johnson transformed CSV; use '-' for stdout. Defaults to processed directory.",
    ),
):
    """
    Apply Yeo-Johnson transform to all numeric columns.
    """
    _run_scaler(apply_yeo_johnson, input_file, output_file, "yeo_johnson")


if __name__ == "__main__":
    app()
