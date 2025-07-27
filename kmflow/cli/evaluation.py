import sys
from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from tqdm import tqdm

import kmflow.config as config
import kmflow.utils.cli_utils as cli_utils
import kmflow.utils.evaluation_utils as evaluation_utils

app = typer.Typer(help="K-Means evaluation metrics CLI.")


@app.command("benchmark")
def benchmark(
    input_dir: Path = typer.Argument(
        ...,
        help="Processed-data root under data/ (e.g. 'processed').",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Where to write the merged benchmark CSV; use '-' for stdout (default).",
    ),
    decimals: int = typer.Option(
        3,
        "--decimals",
        "-d",
        help="Decimal places to round metric values to, default is to the nearest thousandths place.",
    ),
):
    """
    Load all calinski & davies CSVs under data/<input_dir>/… and merge into one table.
    """
    processed_root = config.DATA_DIR / input_dir

    with tqdm(total=4, desc="Benchmark", colour="green") as pbar:
        # 1) load calinski
        calinski_df = evaluation_utils.load_calinski_results(processed_root)
        pbar.update(1)

        # 2) load davies
        davies_df = evaluation_utils.load_davies_results(processed_root)
        pbar.update(1)

        # 3) merge & round
        merged = evaluation_utils.merge_benchmarks(calinski_df, davies_df)
        merged["calinski"] = merged["calinski"].round(decimals)
        merged["davies"] = merged["davies"].round(decimals)
        pbar.update(1)

        # 4) write out
        if output_file is None or output_file == Path("-"):
            merged.to_csv(sys.stdout.buffer, index=False)
            logger.success("Benchmark table written to stdout.")
        else:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            merged.to_csv(output_file, index=False)
            logger.success(f"Benchmark table saved to {output_file!r}")
        pbar.update(1)


@app.command("inertia")
def inertia(
    input_file: Path = typer.Argument(..., help="CSV to read (use '-' for stdin)."),
    start: int = typer.Option(1, "--start", "-start", help="Minimum k (inclusive)."),
    stop: int = typer.Option(20, "--stop", "-stop", help="Maximum k (inclusive)."),
    random_state: int = typer.Option(
        4572, "--seed", "-seed", help="Random seed for reproducibility."
    ),
    n_init: int = typer.Option(50, "--n-init", "-n-init", help="Number of initializations per k."),
    algorithm: str = typer.Option(
        "lloyd", "--algorithm", "-algo", help="KMeans algorithm: 'lloyd' or 'elkan'."
    ),
    init: str = typer.Option(
        "k-means++", "--init", "-init", help="Initialization: 'k-means++' or 'random'."
    ),
    numeric_cols: str = typer.Option(
        None,
        "--numeric-cols",
        "-nc",
        help="Comma-separated list of numeric columns; omit to use all numeric columns.",
        callback=lambda x: cli_utils.comma_split(x) if isinstance(x, str) else x,
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Where to write inertia CSV; use '-' for stdout (default).",
    ),
):
    """
    Compute K-Means inertia over k=start…stop.
    """
    # 1) load
    df = cli_utils.read_df(input_file)

    if not numeric_cols:
        numeric_cols_arg = None
    else:
        numeric_cols_arg = numeric_cols

    # 2) compute
    ks = tqdm(range(start, stop + 1), desc="Inertia", colour="green")
    inertia_df = evaluation_utils.compute_inertia_scores(
        df=df,
        k_range=ks,
        numeric_cols=numeric_cols_arg,
        random_state=random_state,
        n_init=n_init,
        algorithm=algorithm,
        init=init,
    )

    # 3) write out
    if output_file is None:
        output_file = Path.cwd() / "stdin_inertia.csv"
    cli_utils._write_df(inertia_df, output_file)


@app.command("silhouette")
def silhouette(
    input_file: Path = typer.Argument(..., help="CSV to read (use '-' for stdin)."),
    random_state: int = typer.Option(
        4572, "--seed", "-seed", help="Random seed for reproducibility."
    ),
    n_init: int = typer.Option(50, "--n-init", "-n-init", help="Number of initializations per k."),
    algorithm: str = typer.Option(
        "lloyd", "--algorithm", "-algo", help="KMeans algorithm: 'lloyd' or 'elkan'."
    ),
    init: str = typer.Option(
        "k-means++", "--init", "-init", help="Initialization: 'k-means++' or 'random'."
    ),
    numeric_cols: str = typer.Option(
        None,
        "--numeric-cols",
        "-nc",
        help="Comma-separated list of numeric columns; omit to use all numeric columns.",
        callback=lambda x: cli_utils.comma_split(x) if isinstance(x, str) else x,
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Where to write silhouette CSV; use '-' for stdout (default).",
    ),
):
    """
    Compute K-Means silhouette score for k=2…n_samples-1.
    """
    # 1) load
    df = cli_utils.read_df(input_file)

    if not numeric_cols:
        numeric_cols_arg = None
    else:
        numeric_cols_arg = numeric_cols

    ks = tqdm(
        range(2, df.select_dtypes(include="number").shape[0]), desc="Silhouette", colour="green"
    )
    silhouette_df = evaluation_utils.compute_silhouette_scores(
        df=df,
        numeric_cols=numeric_cols_arg,
        k_values=ks,
        random_state=random_state,
        n_init=n_init,
        algorithm=algorithm,
        init=init,
    )

    if output_file is None:
        output_file = Path.cwd() / "stdin_silhouette.csv"
    cli_utils._write_df(silhouette_df, output_file)


@app.command("calinski")
def calinski(
    input_file: Path = typer.Argument(..., help="CSV to read (use '-' for stdin)."),
    random_state: int = typer.Option(
        4572, "--seed", "-seed", help="Random seed for reproducibility."
    ),
    n_init: int = typer.Option(50, "--n-init", "-n-init", help="Number of initializations per k."),
    algorithm: str = typer.Option(
        "lloyd", "--algorithm", "-algo", help="KMeans algorithm: 'lloyd' or 'elkan'."
    ),
    init: str = typer.Option(
        "k-means++", "--init", "-init", help="Initialization: 'k-means++' or 'random'."
    ),
    numeric_cols: str = typer.Option(
        None,
        "--numeric-cols",
        "-nc",
        help="Comma-separated list of numeric columns; omit to use all numeric columns.",
        callback=lambda x: cli_utils.comma_split(x) if isinstance(x, str) else x,
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Where to write Calinski-Harabasz CSV; use '-' for stdout (default).",
    ),
):
    """
    Compute K-Means Calinski-Harabasz score for k=2…n_samples-1.
    """
    # 1) load
    df = cli_utils.read_df(input_file)

    if not numeric_cols:
        numeric_cols_arg = None
    else:
        numeric_cols_arg = numeric_cols

    ks = tqdm(
        range(2, df.select_dtypes(include="number").shape[0]), desc="Calinski", colour="green"
    )
    calinski_df = evaluation_utils.compute_calinski_scores(
        df=df,
        numeric_cols=numeric_cols_arg,
        k_values=ks,
        random_state=random_state,
        n_init=n_init,
        algorithm=algorithm,
        init=init,
    )

    if output_file is None:
        output_file = Path.cwd() / "stdin_calinski.csv"
    cli_utils._write_df(calinski_df, output_file)


@app.command("davies")
def davies(
    input_file: Path = typer.Argument(..., help="CSV to read (use '-' for stdin)."),
    random_state: int = typer.Option(
        4572, "--seed", "-seed", help="Random seed for reproducibility."
    ),
    n_init: int = typer.Option(50, "--n-init", "-n-init", help="Number of initializations per k."),
    algorithm: str = typer.Option(
        "lloyd", "--algorithm", "-algo", help="KMeans algorithm: 'lloyd' or 'elkan'."
    ),
    init: str = typer.Option(
        "k-means++", "--init", "-init", help="Initialization: 'k-means++' or 'random'."
    ),
    numeric_cols: str = typer.Option(
        None,
        "--numeric-cols",
        "-nc",
        help="Comma-separated list of numeric columns; omit to use all numeric columns.",
        callback=lambda x: cli_utils.comma_split(x) if isinstance(x, str) else x,
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Where to write Davies-Bouldin CSV; use '-' for stdout (default).",
    ),
):
    """
    Compute K-Means Davies-Bouldin score for k=2…n_samples-1.
    """
    # 1) load
    df = cli_utils.read_df(input_file)

    if not numeric_cols:
        numeric_cols_arg = None
    else:
        numeric_cols_arg = numeric_cols

    ks = tqdm(range(2, df.select_dtypes(include="number").shape[0]), desc="Davies", colour="green")
    davies_df = evaluation_utils.compute_davies_scores(
        df=df,
        numeric_cols=numeric_cols_arg,
        k_values=ks,
        random_state=random_state,
        n_init=n_init,
        algorithm=algorithm,
        init=init,
    )

    if output_file is None:
        output_file = Path.cwd() / "stdin_davies.csv"
    cli_utils._write_df(davies_df, output_file)


if __name__ == "__main__":
    app()
