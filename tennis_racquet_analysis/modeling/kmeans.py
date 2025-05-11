import typer
from loguru import logger
from pathlib import Path
from typing import List
from tqdm import tqdm

from tennis_racquet_analysis.config import DATA_DIR, PROCESSED_DATA_DIR
from tennis_racquet_analysis.preprocessing_utils import load_data
from tennis_racquet_analysis.modeling.kmeans_utils import (
    compute_inertia_scores,
    compute_silhouette_scores,
    fit_kmeans,
    batch_kmeans,
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
    progress_bar = (
        tqdm(
            range(start, stop + 1),
            desc="Inertia",
            ncols=100,
        ),
    )
    inertia_df = (
        compute_inertia_scores(
            df,
            feature_columns if feature_columns else None,
            progress_bar,
        ),
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
    progress_bar = (
        tqdm(
            range(start, stop + 1),
            desc="Silhouette",
            ncols=100,
        ),
    )
    silhouette_df = compute_silhouette_scores(
        df,
        feature_columns if feature_columns else None,
        progress_bar,
    )
    stem = Path(input_file).stem
    output_filename = f"{stem}_silhouette.csv"
    write_csv(silhouette_df, prefix=stem, suffix="silhouette", output_dir=output_dir)
    logger.success(f"Inertia results saved to {(output_dir / output_filename)!r}")


@app.command("cluster")
def km_cluster(
    input_file: str = typer.Argument(..., help="csv filename under processed data/"),
    dir_label: str = typer.Argument(..., help="Sub-folder under data/"),
    k: int = typer.Option(
        ...,
        "--k",
        "-k",
        help="Number of clusters to fit",
    ),
    feature_columns: List[str] = typer.Option(
        None,
        "--feature-column",
        "-f",
        help="Which numeric columns to use; repeat to supply mulitple. Defaults to all numeric.",
    ),
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR,
        "--output-dir",
        "-o",
        exists=True,
        dir_okay=True,
        file_okay=False,
        help="Directory to write the labeled csv (default: data/processed).",
    ),
):
    input_path = DATA_DIR / dir_label / input_file
    df = load_data(input_path)
    steps = tqdm(total=2, desc="Clustering", ncols=100)
    df_labeled = fit_kmeans(
        df,
        k=k,
        feature_columns=feature_columns if feature_columns else None,
        label_column="cluster",
    )
    steps.update(1)
    stem = Path(input_file).stem
    output_filename = f"{stem}_clustered_{k}.csv"
    write_csv(df_labeled, prefix=stem, suffix=f"clustered_{k}", output_dir=output_dir)
    steps.update(1)
    steps.close()
    logger.success(f"Clustered data saved to {(output_dir / output_filename)!r}")


@app.command("batch-cluster")
def batch_cluster_export(
    input_file: str = typer.Argument(..., help="csv filename under data subfolder."),
    dir_label: str = typer.Argument(..., help="Sub-folder under data/"),
    start: int = typer.Option(
        1,
        "--start",
        "-s",
        help="Minimum k (inclusive).",
    ),
    stop: int = typer.Option(
        10,
        "--stop",
        "-e",
        help="Maximum k (inclusive).",
    ),
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR,
        "--output-dir",
        "-o",
        exists=True,
        dir_okay=True,
        file_okay=False,
        help="Where to write the labeled csv.",
    ),
):
    df = load_data(DATA_DIR / dir_label / input_file)
    progress_bar = tqdm(range(start, stop + 1), desc="Batch Clustering:", ncols=100)
    df_labeled = batch_kmeans(df, k_range=progress_bar)
    prefix = Path(input_file).stem
    suffix = "clusters_" + "_".join(str(k) for k in range(start, stop + 1))
    output_path = write_csv(df_labeled, prefix=prefix, suffix=suffix, output_dir=output_dir)
    logger.success(f"Saved batch clusters for k={start}-{stop} -> {output_path!r}")


if __name__ == "__main__":
    app()
