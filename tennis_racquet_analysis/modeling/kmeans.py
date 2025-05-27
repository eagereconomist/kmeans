import typer
from loguru import logger
from pathlib import Path
from typing import List
from tqdm import tqdm

from tennis_racquet_analysis.config import DATA_DIR
from tennis_racquet_analysis.preprocessing_utils import load_data
from tennis_racquet_analysis.modeling.kmeans_utils import (
    fit_kmeans,
    batch_kmeans,
)
from tennis_racquet_analysis.processing_utils import write_csv

app = typer.Typer()


@app.command("fit-kmeans")
def km_cluster(
    input_file: str = typer.Argument(..., help="csv filename under processed data/"),
    input_dir: str = typer.Option(
        "processed",
        "--input-dir",
        "-d",
        help="Sub-folder under data/ (e.g. external, interim, processed, raw), where the input file lives.",
    ),
    k: int = typer.Option(
        ...,
        "--k",
        "-k",
        help="Number of clusters to fit",
    ),
    random_state: int = typer.Option(4572, "--seed", help="Random seed for reproducibility."),
    n_init: int = typer.Option(
        50, "--n-init", "-n", help="Number of times kmeans is run with differnet centroid seeds."
    ),
    algorithm: str = typer.Option(
        "lloyd", "--algorithm", "-a", help="KMeans algorithm: 'lloyd' or 'elkan'."
    ),
    init: str = typer.Option(
        "k-means++", "--init", "-i", help="Initialization: 'k-means++' or 'random'"
    ),
    feature_columns: List[str] = typer.Option(
        None,
        "--feature-column",
        "-f",
        help="Which numeric columns to use; repeat to supply mulitple. Defaults to all numeric.",
    ),
    output_dir: str = typer.Option(
        "processed",
        "--output-dir",
        "-o",
        help="Sub-folder under data/ (e.g. external, interim, processed, raw), where the file will be output.",
    ),
):
    input_path = DATA_DIR / input_dir / input_file
    df = load_data(input_path)
    output_path = DATA_DIR / output_dir
    steps = tqdm(total=2, desc="Clustering")
    df_labeled = fit_kmeans(
        df,
        k=k,
        feature_columns=feature_columns,
        random_state=random_state,
        n_init=n_init,
        algorithm=algorithm,
        init=init,
        label_column="cluster",
    )
    steps.update(1)
    stem = Path(input_file).stem
    write_csv(df_labeled, prefix=stem, suffix=f"clustered_{k}", output_dir=output_path)
    steps.update(1)
    steps.close()
    logger.success(f"Saved Clustered Data -> {(output_dir / output_path)!r}")


@app.command("batch-kmeans")
def batch_kmeans_export(
    input_file: str = typer.Argument(..., help="csv filename under data subfolder."),
    input_dir: str = typer.Option(
        "processed",
        "--input-dir",
        "-d",
        help="Sub-folder under data/ (e.g. external, interim, processed, raw), where the input file lives.",
    ),
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
    random_state: int = typer.Option(4572, "--seed", help="Random seed for reproducibility."),
    n_init: int = typer.Option(
        50, "--n-init", "-n", help="Number of times kmeans is run with differnet centroid seeds."
    ),
    algorithm: str = typer.Option(
        "lloyd", "--algorithm", "-a", help="KMeans algorithm: 'lloyd' or 'elkan'."
    ),
    init: str = typer.Option(
        "k-means++", "--init", "-i", help="Initialization: 'k-means++' or 'random'"
    ),
    output_dir: str = typer.Option(
        "processed",
        "--output-dir",
        "-o",
        help="Sub-folder under data/ (e.g. external, interim, processed, raw), where the file will be output.",
    ),
):
    df = load_data(DATA_DIR / input_dir / input_file)
    output_path = DATA_DIR / output_dir
    progress_bar = tqdm(range(start, stop + 1), desc="Batch Clustering")
    df_labeled = batch_kmeans(
        df,
        k_range=progress_bar,
        init=init,
        n_init=n_init,
        random_state=random_state,
        algorithm=algorithm,
    )
    prefix = Path(input_file).stem
    suffix = "".join(f"{start}-{stop}")
    write_csv(df_labeled, prefix=prefix, suffix=suffix, output_dir=output_path)
    logger.success(f"Saved Batch Clusters for k={start}-{stop} -> {output_path!r}")


if __name__ == "__main__":
    app()
