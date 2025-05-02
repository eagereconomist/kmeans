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
        help="Directory where processed csv's will be written.",
    ),
    prefix: str = typer.Option(
        None,
        "--prefix",
        "-p",
        help="By default, `prefix` is None, but there is the option to pick a `prefix. "
        "Base name for the output files (before the scaling suffix).",
    ),
):
    """
    Load a pre-feature-engineered csv, apply several scalers,
    and write one output csv per scaler.
    """
    input_path = input_dir / input_file
    logger.info(f"Loading feature-engineered data from {input_path!r}")
    df = load_data(input_path)
    if prefix is None:
        prefix = Path(input_file).stem
    if not isinstance(prefix, str):
        prefix = Path(input_file).stem
    steps = [
        ("normalized", apply_normalizer, "normalized"),
        ("standardized", apply_standardization, "standardized"),
        ("minmax", apply_minmax, "minmax"),
        ("log_transformed", log_transform, "log_scale"),
        ("yeo_johnson", yeo_johnson, "yeo_johnson"),
    ]

    processed_paths = {}

    for step_name, scaler_func, suffix in tqdm(steps, desc="Scaling Steps", ncols=100):
        logger.info(f"Applying '{step_name}' scaler...")
        df_scaled = scaler_func(df)
        output_path = write_csv(
            df_scaled,
            prefix=prefix,
            suffix=suffix,
            output_dir=output_dir,
        )
        processed_paths[step_name] = output_path
        logger.success(f"-> Wrote {step_name} to {output_path!r}")
    logger.info(f"Processed DataFrame type: {type(df)}")
    logger.info(f"Processed DataFrame dimensions: {df.shape}")
    typer.echo(f"All done. Files written to {output_dir}")
    return processed_paths


if __name__ == "__main__":
    app()
