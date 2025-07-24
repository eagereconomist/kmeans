from pathlib import Path

import typer

import kmflow.utils.process_utils as process_utils
import kmflow.config as config

app = typer.Typer(help="Apply individual scaler to a preprocessed CSV.")


@app.command("norm")
def normalize(
    input_file: Path = typer.Argument(..., help="CSV to read (use '-' to read from stdin)."),
    output_file: Path = typer.Option(
        config.PROCESSED_DATA_DIR,
        "--output-file",
        "-o",
        help="Where to save normalized CSV; use '-' for stdout. Defaults to processed directory.",
    ),
):
    """L2-normalize all numeric columns."""
    process_utils._run_scaler_with_progress(
        process_utils.apply_normalizer,
        input_file,
        output_file,
        "norm",
        desc="Applying Normalize Scaler to Data",
    )


@app.command("std")
def standardize(
    input_file: Path = typer.Argument(
        ...,
        help="CSV to read (use '-' to read from stdin).",
    ),
    output_file: Path = typer.Option(
        config.PROCESSED_DATA_DIR,
        "--output-file",
        "-o",
        help="Where to save standardized CSV; use '-' for stdout. Defaults to processed directory.",
    ),
):
    """
    Z-score standardize all numeric columns.
    """
    process_utils._run_scaler_with_progress(
        process_utils.apply_standardization,
        input_file,
        output_file,
        "std",
        desc="Applying Standardize Scaler to Data",
    )


@app.command("minmax")
def minmax(
    input_file: Path = typer.Argument(
        ...,
        help="CSV to read (use '-' to read from stdin).",
    ),
    output_file: Path = typer.Option(
        config.PROCESSED_DATA_DIR,
        "--output-file",
        "-o",
        help="Where to save min-max scaled CSV; use '-' for stdout. Defaults to processed directory.",
    ),
):
    """
    Scale all numeric columns to [0,1].
    """
    process_utils._run_scaler_with_progress(
        process_utils.apply_minmax,
        input_file,
        output_file,
        "minmax",
        desc="Applying MinMax Scaler to Data",
    )


@app.command("log1p")
def log_scale(
    input_file: Path = typer.Argument(
        ...,
        help="CSV to read (use '-' to read from stdin).",
    ),
    output_file: Path = typer.Option(
        config.PROCESSED_DATA_DIR,
        "--output-file",
        "-o",
        help="Where to save log1p-transformed CSV; use '-' for stdout. Defaults to processed directory.",
    ),
):
    """
    Apply log(1 + x) transform to all numeric columns.
    """
    process_utils._run_scaler_with_progress(
        process_utils.apply_log1p,
        input_file,
        output_file,
        "log1p",
        desc="Applying log(1 + x) Scaler to Data",
    )


@app.command("yj")
def yeo_johnson_scale(
    input_file: Path = typer.Argument(
        ...,
        help="CSV to read (use '-' to read from stdin).",
    ),
    output_file: Path = typer.Option(
        config.PROCESSED_DATA_DIR,
        "--output-file",
        "-o",
        help="Where to save Yeo-Johnson transformed CSV; use '-' for stdout. Defaults to processed directory.",
    ),
):
    """
    Apply Yeo-Johnson transform to all numeric columns.
    """
    process_utils._run_scaler_with_progress(
        process_utils.apply_yeo_johnson,
        input_file,
        output_file,
        "yj",
        desc="Applying Yeo-Johnson Scaler to Data",
    )


if __name__ == "__main__":
    app()
