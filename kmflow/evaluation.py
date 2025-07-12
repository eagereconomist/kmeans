import typer
from loguru import logger
from pathlib import Path
from typing import List
from tqdm import tqdm

from kmflow.config import DATA_DIR
from kmflow.preprocess_utils import load_data
from kmflow.process_utils import write_csv
from kmflow.evaluation_utils import (
    compute_inertia_scores,
    compute_silhouette_scores,
    compute_calinski_scores,
    compute_davies_scores,
    load_calinski_results,
    load_davies_results,
    merge_benchmarks,
)

app = typer.Typer()


@app.command("benchmark")
def benchmark(
    input_dir: Path = typer.Option(
        "processed",
        "--input-dir",
        "-d",
        help="Sub-folder under data/ (e.g. external, interim, processed, raw), where the input file lives.",
    ),
    output_file: Path = typer.Option(
        None,
        "--output-file",
        "-o",
        writable=True,
        dir_okay=False,
        help="Optional path to write the benchmark table as csv",
    ),
    decimals: int = typer.Option(
        3,
        "--decimals",
        "-d",
        help="Number of decimal places to round metric values to",
    ),
):
    processed_root = DATA_DIR / input_dir
    calinski_df = load_calinski_results(processed_root)
    davies_df = load_davies_results(processed_root)

    df = merge_benchmarks(calinski_df, davies_df)
    for col in ("calinski", "davies"):
        df[col] = df[col].round(decimals)

    if output_file:
        df.to_csv(output_file, index=False)
        typer.echo(f"Benchmark table saved to {output_file}")


@app.command("inertia")
def km_inertia(
    input_file: str = typer.Argument(..., help="csv filename under the data subfolder."),
    input_dir: str = typer.Option(
        "processed",
        "--input-dir",
        "-d",
        help="Sub-folder under data/ (e.g. external, interim, processed, raw), where the input file lives.",
    ),
    start: int = typer.Option(1, "--start", "-s", help="Minimum k (inclusive)."),
    stop: int = typer.Option(20, "--stop", "-e", help="Maximum k (inclusive)."),
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
    progress_bar = tqdm(range(start, stop + 1), desc="Inertia")
    inertia_df = compute_inertia_scores(
        df=df,
        feature_columns=feature_columns,
        k_range=progress_bar,
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
    progress_bar = tqdm(range(2, 21), desc="Silhouette")
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


@app.command("calinski")
def km_calinski(
    input_file: str = typer.Argument(..., help="csv filename under the data subfolder."),
    input_dir: str = typer.Option(
        "processed",
        "--input-dir",
        "-d",
        help="Sub-folder under data/ (e.g. external, interim, processed, raw), where the input file lives.",
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
    progress_bar = tqdm(range(2, 21), desc="Calinski-Harabasz")
    calinski_df = compute_calinski_scores(
        df=df,
        feature_columns=feature_columns,
        k_values=progress_bar,
        random_state=random_state,
        n_init=n_init,
        algorithm=algorithm,
        init=init,
    )
    stem = Path(input_file).stem
    write_csv(calinski_df, prefix=stem, suffix="calinski", output_dir=output_path)
    logger.success(f"Saved Calinski-Harabasz Scores -> {(output_dir / output_path)!r}")


@app.command("davies")
def km_davies(
    input_file: str = typer.Argument(..., help="csv filename under the data subfolder."),
    input_dir: str = typer.Option(
        "processed",
        "--input-dir",
        "-d",
        help="Sub-folder under data/ (e.g. external, interim, processed, raw), where the input file lives.",
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
    progress_bar = tqdm(range(2, 21), desc="Davies-Bouldin")
    davies_df = compute_davies_scores(
        df=df,
        feature_columns=feature_columns,
        k_values=progress_bar,
        random_state=random_state,
        n_init=n_init,
        algorithm=algorithm,
        init=init,
    )
    stem = Path(input_file).stem
    write_csv(davies_df, prefix=stem, suffix="davies", output_dir=output_path)
    logger.success(f"Saved Davies-Bouldin Scores -> {(output_dir / output_path)!r}")


if __name__ == "__main__":
    app()
