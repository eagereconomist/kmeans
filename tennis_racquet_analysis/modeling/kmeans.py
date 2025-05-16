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
    input_dir: str = typer.Option(
        "processed",
        "--input-dir",
        "-d",
        help="Sub-folder under data/ (e.g. external, interim, processed, raw), where the input file lives.",
    ),
    random_state: int = typer.Option(
        4572, "--seed", "-s", help="Random seed for reproducibility."
    ),
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
        help="Name of numeric column to include; repeat flag to add more."
        "Defaults to all numeric columns.",
    ),
    output_dir: str = typer.Option(
        "processed",
        "--output-dir",
        "-o",
        help="Sub-folder under data/ (e.g. external, interim, processed, raw), where the file will be output.",
    ),
):
    input_path = DATA_DIR / input_dir / input_file
    output_path = DATA_DIR / output_dir
    df = load_data(input_path)
    n_samples = df.shape[0]
    progress_bar = tqdm(range(1, n_samples + 1), desc="Inertia", ncols=100)
    inertia_df = compute_inertia_scores(
        df=df,
        feature_columns=feature_columns,
        k_values=progress_bar,
        random_state=random_state,
        n_init=n_init,
        algorithm=algorithm,
        init=init,
    )
    stem = Path(input_file).stem
    write_csv(inertia_df, prefix=stem, suffix="inertia", output_dir=output_path)
    logger.success(f"Saved Inertia Scores -> {(output_dir / output_path)!r}")


@app.command("silhouette")
def km_silhouette(
    input_file: str = typer.Argument(..., help="csv filename under the data subfolder."),
    input_dir: str = typer.Option(
        "processed",
        "--input-dir",
        "-d",
        help="Sub-folder under data/ (e.g. external, interim, processed, raw), where the input file lives.",
    ),
    random_state: int = typer.Option(
        4572, "--seed", "-s", help="Random seed for reproducibility."
    ),
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
        help="Name of numeric column to include; repeat flag to add more."
        "Defaults to all numeric columns.",
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
    n_samples = df.shape[0]
    progress_bar = tqdm(range(2, n_samples), desc="Silhouette", ncols=100)
    silhouette_df = compute_silhouette_scores(
        df=df,
        feature_columns=feature_columns,
        k_values=progress_bar,
        random_state=random_state,
        n_init=n_init,
        algorithm=algorithm,
        init=init,
    )
    stem = Path(input_file).stem
    write_csv(silhouette_df, prefix=stem, suffix="silhouette", output_dir=output_path)
    logger.success(f"Saved Silhouette Scores -> {(output_dir / output_path)!r}")


@app.command("cluster")
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
    random_state: int = typer.Option(
        4572, "--seed", "-s", help="Random seed for reproducibility."
    ),
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
    steps = tqdm(total=2, desc="Clustering", ncols=100)
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


@app.command("batch-cluster")
def batch_cluster_export(
    input_file: str = typer.Argument(..., help="csv filename under data subfolder."),
    input_dir: Path = typer.Option(
        PROCESSED_DATA_DIR,
        "--input_dir",
        "-d",
        exists=True,
        dir_okay=True,
        file_okay=True,
        help="Directory where files live.",
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
    random_state: int = typer.Option(
        4572, "--seed", "-s", help="Random seed for reproducibility."
    ),
    n_init: int = typer.Option(
        50, "--n-init", "-n", help="Number of times kmeans is run with differnet centroid seeds."
    ),
    algorithm: str = typer.Option(
        "lloyd", "--algorithm", "-a", help="KMeans algorithm: 'lloyd' or 'elkan'."
    ),
    init: str = typer.Option(
        "k-means++", "--init", "-i", help="Initialization: 'k-means++' or 'random'"
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
    df = load_data(DATA_DIR / input_dir / input_file)
    progress_bar = tqdm(range(start, stop + 1), desc="Batch Clustering:", ncols=100)
    df_labeled = batch_kmeans(
        df,
        k_range=progress_bar,
        init=init,
        n_init=n_init,
        random_state=random_state,
        algorithm=algorithm,
    )
    prefix = Path(input_file).stem
    suffix = "clusters_" + "_".join(str(k) for k in range(start, stop + 1))
    output_path = write_csv(df_labeled, prefix=prefix, suffix=suffix, output_dir=output_dir)
    logger.success(f"Saved Batch Clusters for k={start}-{stop} -> {output_path!r}")


if __name__ == "__main__":
    app()
