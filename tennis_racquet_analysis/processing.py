from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer

from tennis_racquet_analysis.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from tennis_racquet_analysis.processing_utils import (
    load_data,
    write_csv,
    apply_normalizer,
    apply_standardization,
    apply_minmax,
    log_transform,
    yeo_johnson,
)

app = typer.Typer()


@app.command()
def main(
    input_file: str = typer.Argument(..., help="Feature-engineered csv filename."),
    input_dir: Path = typer.Option(
        INTERIM_DATA_DIR,
        "--input_dir",
        "-d",
        exists=True,
        dir_okay=True,
        file_okay=True,
        help="Directory where feature-engineered files live.",
    ),
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR,
        "--output-dir",
        "-o",
        exists=True,
        dir_okay=True,
        file_okay=False,
        help="Where processed csv's will be written.",
    ),
    prefix: str = typer.Option(
        "tennis_racquets",
        "--prefix",
        "-p",
        help="Base name for the output files (before the scaling suffix).",
    ),
):
    """
    Load a pre-feature-engineered csv, apply several scalers,
    and write one output csv per scaler.
    """
    input_path = input_dir / input_file
    logger.info(f"Loading feature-engineered data from {input_path!r}")
    df = load_data(input_path)
    stem = Path(input_file).stem
    steps = [
        ("normalized", apply_normalizer, "normalized"),
        ("standardized", apply_standardization, "standardized"),
        ("minmax", apply_minmax, "minmax"),
        ("log_transformed", log_transform, "log_scale"),
        ("yeo_johnson", yeo_johnson, "yeo_johnson"),
    ]

    out_paths = {}

    for name, func in tqdm(steps, desc="Scaling Steps", ncols=100):
        logger.info(f"Applying '{name}' scaler")
        df_scaled = func(df)
        file_name = f"{prefix}_{stem}_{name}.csv"
        destination = output_dir / file_name
        df_scaled.to_csv(destination, index=False)
        logger.success(f"Wrote {name} -> {destination!r}")
        out_paths[name] = destination
        typer.echo(f"All done - files written to {output_dir}")
        return out_paths


if __name__ == "__main__":
    app()
