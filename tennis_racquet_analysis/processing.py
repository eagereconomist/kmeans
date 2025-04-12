from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer

from tennis_racquet_analysis.config import INTERIM_DATA_DIR
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
    input_path: Path = INTERIM_DATA_DIR / "tennis_racquets_features.csv",
):
    logger.info("Loading preprocessed dataset...")
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
        scaling_steps, total=len(scaling_steps), desc="Scaling Steps:"
    ):
        logger.info(f"Applying scaling: {scaling_name}")
        df_processed = scaling_func(df_preprocessed)
        processed_results[scaling_name] = write_csv(df_processed, "processed", file_label)

    logger.success("Data processing complete!")
    typer.echo("Processed DataFrames: " + ", ".join(processed_results.keys()))
    return processed_results


if __name__ == "__main__":
    app()
