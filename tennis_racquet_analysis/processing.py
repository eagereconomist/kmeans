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
    input_file: str = typer.Argument(..., help="Preprocessed Feature-Engineered csv filename."),
    input_dir: Path = typer.Option(
        INTERIM_DATA_DIR,
        "--input_dir",
        "-d",
        exists=True,
        file_okay=True,
        help="Directory where preprocessed feature-engineered files live.",
    ),
    file_label: str = typer.Option(
        "processed",
        "--label",
        "-l",
        help="Suffix for the output file before .csv",
    ),
):
    input_path = input_dir / input_file
    logger.info("Loading preprocessed feature-engineered dataset...")
    df_preprocessed = load_data(input_path)
    scaling_steps = [
        ("normalized", apply_normalizer, "normalized"),
        ("standardized", apply_standardization, "standardized"),
        ("minmax", apply_minmax, "minmax"),
        ("log_transformed", log_transform, "log_scale"),
        ("yeo_johnson", yeo_johnson, "yeo_johnson"),
    ]

    processed_results = {}

    for scaling_name, scaling_func, file_label in tqdm(
        scaling_steps, total=len(scaling_steps), ncols=100, desc="Scaling Steps"
    ):
        logger.info(f"Applying scaling: {scaling_name}")
        df_processed = scaling_func(df_preprocessed)
        processed_results[scaling_name] = write_csv(df_processed, file_label)

    logger.success("Data processing complete!")
    typer.echo("Wrote processed files to " + str(PROCESSED_DATA_DIR))
    return processed_results


if __name__ == "__main__":
    app()
