import typer
from pathlib import Path
from typing import List, Optional

from kmflow.cli_utils import read_df, _write_df
from kmflow.modeling.kmeans_utils import fit_kmeans, batch_kmeans

app = typer.Typer(help="K-means clustering commands.")


@app.command("fit-km")
def fit_km_cli(
    input_file: Path = typer.Argument(..., help="CSV to read (use '-' for stdin)."),
    k: int = typer.Option(..., "--k", "-k", help="Number of clusters to fit."),
    random_state: int = typer.Option(4572, "--seed", "-seed", help="Random seed."),
    n_init: int = typer.Option(50, "--n-init", "-n-init", help="Runs with different seeds."),
    algorithm: str = typer.Option("lloyd", "--algorithm", "-algo", help="'lloyd' or 'elkan'."),
    init: str = typer.Option("k-means++", "--init", "-init", help="Init method."),
    numeric_cols: Optional[List[str]] = typer.Option(
        None,
        "--numeric-cols",
        "-numeric-cols",
        help="Numeric cols to use; repeat for multiple.",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Where to save; use '-' for stdout. Defaults to ./<stem>_clustered_<k>.csv",
    ),
):
    df = read_df(input_file)
    df_out = fit_kmeans(
        df=df,
        k=k,
        numeric_cols=numeric_cols,
        init=init,
        n_init=n_init,
        random_state=random_state,
        algorithm=algorithm,
        cluster_col="cluster",
    )

    stem = input_file.stem if input_file != Path("-") else "stdin"
    out = output_file or Path.cwd() / f"{stem}_clustered_{k}.csv"
    _write_df(df_out, out)


@app.command("batch-km")
def batch_km_cli(
    input_file: Path = typer.Argument(..., help="CSV to read (use '-' for stdin)."),
    start: int = typer.Option(1, "--start", "-start", help="Minimum k."),
    stop: int = typer.Option(10, "--stop", "-stop", help="Maximum k."),
    random_state: int = typer.Option(4572, "--seed", "-seed", help="Random seed."),
    n_init: int = typer.Option(50, "--n-init", "-n-init", help="Runs per k."),
    algorithm: str = typer.Option("lloyd", "--algorithm", "-algo", help="'lloyd' or 'elkan'."),
    init: str = typer.Option("k-means++", "--init", "-init", help="Init method."),
    numeric_cols: Optional[List[str]] = typer.Option(
        None,
        "--numeric-cols",
        "-numeric-cols",
        help="Numeric cols to use; repeat for multiple.",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Where to save; use '-' for stdout. Defaults to ./<stem>_batch_<start>-<stop>.csv",
    ),
):
    df = read_df(input_file)
    df_out = batch_kmeans(
        df=df,
        k_range=range(start, stop + 1),
        numeric_cols=numeric_cols,
        init=init,
        n_init=n_init,
        random_state=random_state,
        algorithm=algorithm,
        cluster_col="cluster",
    )

    stem = input_file.stem if input_file != Path("-") else "stdin"
    suffix = f"{start}-{stop}"
    out = output_file or Path.cwd() / f"{stem}_batch_{suffix}.csv"
    _write_df(df_out, out)


if __name__ == "__main__":
    app()
