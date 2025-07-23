from pathlib import Path
from typing import List, Optional

import typer
from loguru import logger
from tqdm import tqdm

import kmflow.utils.cli_utils as cli_utils
import kmflow.utils.plots_utils as plots_utils
import kmflow.utils.kmeans_utils as kmeans_utils

app = typer.Typer(help="K-Means clustering commands.")


@app.command("fit-km")
def fit_km_cli(
    input_file: Path = typer.Argument(..., help="CSV to read (use '-' for stdin)."),
    k: int = typer.Argument(..., help="Number of clusters to fit."),
    random_state: int = typer.Option(4572, "--seed", "-seed", help="Random seed."),
    n_init: int = typer.Option(50, "--n-init", "-n-init", help="Runs with different seeds."),
    algorithm: str = typer.Option("lloyd", "--algorithm", "-algo", help="'lloyd' or 'elkan'."),
    init: str = typer.Option("k-means++", "--init", "-init", help="Init method."),
    numeric_cols: Optional[List[str]] = typer.Option(
        None, "--numeric-cols", "-numeric-cols", help="Numeric cols to use; repeat for multiple."
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Where to save; use '-' for stdout. Defaults to ./<stem>_clustered_<k>.csv",
    ),
):
    """
    Fit K-means with k clusters.
    """
    stem = input_file.stem if input_file != Path("-") else "stdin"
    default_name = f"{stem}_clustered_{k}.csv"
    out_path = output_file or (Path.cwd() / default_name)
    if out_path != Path("-"):
        out_path = plots_utils._ensure_unique_path(out_path)

    with tqdm(total=3, desc="Fit K-Means", colour="green") as pbar:
        # 1) load
        df = cli_utils.read_df(input_file)
        pbar.update(1)

        # 2) fit
        df_out = kmeans_utils.fit_kmeans(
            df=df,
            k=k,
            numeric_cols=numeric_cols,
            init=init,
            n_init=n_init,
            random_state=random_state,
            algorithm=algorithm,
            cluster_col="cluster",
        )
        pbar.update(1)

        # 3) write
        cli_utils._write_df(df_out, out_path)
        if out_path == Path("-"):
            logger.success("K-means clustering CSV written to stdout.")
        else:
            logger.success(f"K-means clustering results saved to {out_path!r}")
        pbar.update(1)


@app.command("batch-km")
def batch_km_cli(
    input_file: Path = typer.Argument(..., help="CSV to read (use '-' for stdin)."),
    start: int = typer.Option(1, "--start", "-start", help="Minimum k."),
    stop: int = typer.Option(20, "--stop", "-stop", help="Maximum k."),
    random_state: int = typer.Option(4572, "--seed", "-seed", help="Random seed."),
    n_init: int = typer.Option(50, "--n-init", "-n-init", help="Runs per k."),
    algorithm: str = typer.Option("lloyd", "--algorithm", "-algo", help="'lloyd' or 'elkan'."),
    init: str = typer.Option("k-means++", "--init", "-init", help="Init method."),
    numeric_cols: Optional[List[str]] = typer.Option(
        None, "--numeric-cols", "-numeric-cols", help="Numeric cols to use; repeat for multiple."
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Where to save; use '-' for stdout. Defaults to ./<stem>_batch_<start>-<stop>.csv",
    ),
):
    """
    Run K-means for k in [start…stop] and append each cluster column.
    """
    stem = input_file.stem if input_file != Path("-") else "stdin"
    suffix = f"{start}-{stop}"
    default_name = f"{stem}_batch_km_{suffix}.csv"
    out_path = output_file or (Path.cwd() / default_name)
    if out_path != Path("-"):
        out_path = plots_utils._ensure_unique_path(out_path)

    with tqdm(total=3, desc="Batch K-Means", colour="green") as pbar:
        # 1) load
        df = cli_utils.read_df(input_file)
        pbar.update(1)

        # 2) batch-fit
        df_out = kmeans_utils.batch_kmeans(
            df=df,
            k_range=range(start, stop + 1),
            numeric_cols=numeric_cols,
            init=init,
            n_init=n_init,
            random_state=random_state,
            algorithm=algorithm,
            cluster_col="cluster",
        )
        pbar.update(1)

        # 3) write
        cli_utils._write_df(df_out, out_path)
        if out_path == Path("-"):
            logger.success("Batch K-means clustering CSV written to stdout.")
        else:
            logger.success(f"Batch K-means clustering results saved to {out_path!r}")
        pbar.update(1)


if __name__ == "__main__":
    app()
