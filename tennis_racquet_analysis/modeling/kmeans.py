import typer
from loguru import logger
from pathlib import Path
from typing import List

from tennis_racquet_analysis.config import DATA_DIR, PROCESSED_DATA_DIR
from tennis_racquet_analysis.preprocessing_utils import load_data
from tennis_racquet_analysis.modeling.kmeans_utils import (
    compute_kmeans_inertia,
    compute_silhouette_scores,
)
from tennis_racquet_analysis.processing_utils import write_csv

app = typer.Typer()


@app.command("inertia")
def km_inertia(
    input_file: str = typer.Argument(..., help="csv filename under the data subfolder."),
    input_dir: Path = typer.Option(
        PROCESSED_DATA_DIR,
        "--input_dir",
        "-d",
        exists=True,
        dir_okay=True,
        file_okay=True,
        help="Directory where feature-engineered files live.",
    ),
    start: int = typer.Option(
        1,
        "--start",
        "-s",
        help="Minimum number of clusters (inclusive).",
    ),
    stop: int = typer.Option(
        10,
        "--stop",
        "-e",
        help="Maximum number of clusters (inclusive).",
    ),
    feature_columns: List[str] = typer.Option(
        None,
        "--feature-column",
        "-f",
        help="Name of numeric column to include; repeat flag to add more."
        "Defaults to all numeric columns.",
    ),
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR,
        "--output-dir",
        "-o",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Directory to write the inertia csv (default: data/processed).",
    ),
):
    input_path = input_dir / input_file
    df = load_data(input_path)
    inertia_df = compute_kmeans_inertia(
        df,
        feature_columns if feature_columns else None,
        (start, stop),
    )
    stem = Path(input_file).stem
    output_filename = f"{stem}_inertia.csv"
    write_csv(inertia_df, prefix=stem, suffix="inertia", output_dir=output_dir)
    logger.success(f"Inertia results saved to {(output_dir / output_filename)!r}")


@app.command("silhouette")
def km_silhouette(
    input_file: str = typer.Argument(..., help="csv filename under the data subfolder."),
    dir_label: str = typer.Argument(..., help="Sub-folder under data/"),
    start: int = typer.Option(
        2,
        "--start",
        "-s",
        help="Minimum number of clusters (inclusive).",
    ),
    stop: int = typer.Option(
        10,
        "--stop",
        "-e",
        help="Maximum number of clusters (inclusive).",
    ),
    feature_columns: List[str] = typer.Option(
        None,
        "--feature-column",
        "-f",
        help="Name of numeric column to include; repeat flag to add more."
        "Defaults to all numeric columns.",
    ),
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR,
        "--output-dir",
        "-o",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Directory to write the inertia csv (default: data/processed).",
    ),
):
    df = load_data(DATA_DIR / dir_label / input_file)
    silhouette_df = compute_silhouette_scores(
        df,
        feature_columns if feature_columns else None,
        (start, stop),
    )
    stem = Path(input_file).stem
    output_filename = f"{stem}_silhouette.csv"
    write_csv(silhouette_df, prefix=stem, suffix="silhouette", output_dir=output_dir)
    logger.success(f"Inertia results saved to {(output_dir / output_filename)!r}")


if __name__ == "__main__":
    app()
