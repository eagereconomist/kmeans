import sys
from pathlib import Path
import typer
from tqdm import tqdm
from kmflow.utils import cli_utils, process_utils, pca_utils

app = typer.Typer()


@app.command("pca")
def run_pca(
    input_file: Path = typer.Argument(
        ...,
        help="CSV file to read (use '-' for stdin).",
    ),
    numeric_cols: str = typer.Option(
        "",
        "--numeric-cols",
        "-nc",
        help="Comma-separated list of numeric columns; omit to use all numeric columns.",
    ),
    n_components: int = typer.Option(
        None,
        "--n-components",
        "-c",
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
    df = cli_utils.read_df(input_file)
    stem = input_file.stem if input_file != Path("-") else "stdin"

    # ─── parse numeric_cols into a list or None ────────────────────
    if numeric_cols.strip() == "":
        numeric_cols_arg = None
    else:
        numeric_cols_arg = cli_utils.comma_split(numeric_cols)

    summary = pca_utils.compute_pca(
        df=df,
        numeric_cols=numeric_cols_arg,
        n_components=n_components,
        random_state=random_state,
    )

    if output_dir is None:
        output_dir = Path(f"{stem}_pca")
    if output_dir != Path("-"):
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
            typer.secho(f"{desc} written to stdout.", fg="green")
        else:
            path = process_utils.write_csv(
                df_out, prefix=stem, suffix=suffix, output_dir=output_dir
            )
            typer.secho(f"Saved {desc} -> {path!r}", fg="green")
