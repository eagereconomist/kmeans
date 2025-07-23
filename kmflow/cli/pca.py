import sys
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from kmflow.process_utils import write_csv
from kmflow.cli_utils import read_df, comma_split
from kmflow.pca_utils import compute_pca


app = typer.Typer(help="Principal Component Analysis.")


@app.command("pca")
def pca_summary(
    input_file: Path = typer.Argument(
        ...,
        help="CSV file to read (use '-' for stdin).",
    ),
    numeric_cols: list[str] = typer.Option(
        [],
        "--numeric-cols",
        "-numeric-cols",
        help="Numeric columns to include; comma-separated or repeatable.",
        callback=lambda x: comma_split(x) if isinstance(x, str) else x,
    ),
    n_components: int = typer.Option(
        None,
        "--n-components",
        "-components",
        help="Number of PCs to compute (defaults to all).",
    ),
    random_state: int = typer.Option(
        4572,
        "--seed",
        "-seed",
        help="Random seed for reproducibility.",
    ),
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        "-o",
        dir_okay=True,
        file_okay=False,
        help="Directory where PCA summary CSVs will be saved.",
    ),
):
    """
    Compute PCA loadings, scores, explained variance, and cumulative variance,
    then write four CSVs with a progress bar.
    """
    df = read_df(input_file)
    stem = input_file.stem if input_file != Path("-") else "stdin"

    if not numeric_cols:
        numeric_cols_arg = None
    else:
        numeric_cols_arg = numeric_cols

    summary = compute_pca(
        df=df,
        numeric_cols=numeric_cols_arg,
        n_components=n_components,
        random_state=random_state,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (
            "PCA Loadings",
            summary["loadings"].reset_index().rename(columns={"index": "component"}),
            "pca_loadings",
        ),
        (
            "PCA Scores",
            summary["scores"].reset_index(drop=True),
            f"pca_scores_{summary['scores'].shape[1]}pc" if n_components else "pca_scores",
        ),
        (
            "Proportion Variance",
            summary["pve"].reset_index().rename(columns={"index": "component"}),
            "pca_proportion_var",
        ),
        (
            "Cumulative Variance",
            summary["cpve"].reset_index().rename(columns={"index": "component"}),
            "pca_cumulative_var",
        ),
    ]

    for desc, df_out, suffix in tqdm(tasks, desc="Writing PCA CSVs", colour="green"):
        if output_dir == Path("-"):
            df_out.to_csv(sys.stdout.buffer, index=False)
            logger.success(f"{desc} written to stdout.")
        else:
            path = write_csv(df_out, prefix=stem, suffix=suffix, output_dir=output_dir)
            logger.success(f"Saved {desc} -> {path!r}")


if __name__ == "__main__":
    app()
