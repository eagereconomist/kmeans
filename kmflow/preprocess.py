import sys
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from kmflow.cli_utils import read_df, comma_split, comma_split_int
from kmflow.preprocess_utils import (
    find_iqr_outliers,
    compute_pca_summary,
    drop_column,
    drop_row,
    dotless_column,
)
from kmflow.process_utils import write_csv
from kmflow.config import INTERIM_DATA_DIR

app = typer.Typer(help="Data preprocessing and PCA-summary commands.")


@app.command("outlier")
def iqr_cli(
    input_file: Path = typer.Argument(
        ...,
        help="CSV file to read (use '-' for stdin).",
    ),
    export_outliers: bool = typer.Option(
        False,
        "--export-outliers",
        "-eo",
        help="Write detected outliers to CSV.",
    ),
    remove_outliers: bool = typer.Option(
        False,
        "--remove-outliers",
        "-ro",
        help="Write cleaned CSV (with outliers removed).",
    ),
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory in which to save output CSVs; use '-' for stdout.",
    ),
):
    """
    Identify IQR-based outliers, report count, and optionally export or remove them.
    """
    df = read_df(input_file)
    out = find_iqr_outliers(df)

    if out.empty:
        logger.info("No IQR-based outliers detected.")
        return

    out_df = out.reset_index().rename(
        columns={"level_0": "row_index", "level_1": "column", 0: "outlier_value"}
    )

    if output_dir is None:
        output_dir = INTERIM_DATA_DIR

    # If the user didn't ask to export or remove, just log a summary and exit
    if not export_outliers and not remove_outliers:
        count = out_df.shape[0]
        logger.info(
            f"{count} IQR outliers detected. "
            "Use --export-outliers (Shorthand: -eo) to save them or --remove-outliers (Shorthand: -ro) to drop them."
        )
        return

    if export_outliers:
        if output_dir == Path("-"):
            out_df.to_csv(sys.stdout.buffer, index=False)
            logger.success("Outliers written to stdout.")
        else:
            path = write_csv(
                out_df,
                prefix=input_file.stem,
                suffix="iqr_outliers",
                output_dir=output_dir,
            )
            logger.success(f"Outliers written to {path!r}")

    if remove_outliers:
        rows = out_df["row_index"].unique().tolist()
        cleaned = drop_row(df, rows)
        if output_dir == Path("-"):
            cleaned.to_csv(sys.stdout.buffer, index=False)
            logger.success("Cleaned data written to stdout.")
        else:
            path = write_csv(
                cleaned,
                prefix=input_file.stem,
                suffix="no_outliers",
                output_dir=output_dir,
            )
            logger.success(f"Cleaned data written to {path!r}")


@app.command("preprocess")
def preprocess(
    input_file: Path = typer.Argument(
        ...,
        help="CSV to read (use '-' for stdin).",
    ),
    dropped_columns: list[str] = typer.Option(
        [],
        "--dropped-column",
        "-dc",
        help="Columns to drop, comma-separated or repeatable.",
        callback=lambda x: comma_split(x) if isinstance(x, str) else x,
    ),
    dotless_columns: list[str] = typer.Option(
        [],
        "--dotless-column",
        "-dot",
        help="Columns whose dots to remove, comma-separated or repeatable.",
        callback=lambda x: comma_split(x) if isinstance(x, str) else x,
    ),
    drop_rows: list[int] = typer.Option(
        [],
        "--dropped-row",
        "-dr",
        help="Row indices to drop, comma-separated or repeatable.",
        callback=lambda x: comma_split_int(x) if isinstance(x, str) else x,
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        "-p",
        help="Show first 5 rows of the final DataFrame.",
    ),
    output_file: Path = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Where to save the cleaned CSV; use '-' for stdout.",
    ),
):
    """
    Apply column/row drops and dotless renaming to a CSV. Does NOT handle IQR outliers.
    """
    df = read_df(input_file)
    stem = input_file.stem if input_file != Path("-") else "stdin"

    steps: list[tuple[str, callable, list]] = []
    for col in dropped_columns:
        steps.append(("drop_column", drop_column, [col]))
    for col in dotless_columns:
        steps.append(("dotless_column", dotless_column, [col]))
    for idx in drop_rows:
        steps.append(("drop_row", drop_row, [[idx]]))

    for name, func, args in tqdm(steps, desc="Data Preprocessing Steps", colour="green"):
        logger.info(f"Applying {name}...")
        df = func(df, *args)

    if preview:
        typer.echo(df.head())

    if output_file is None:
        output_file = INTERIM_DATA_DIR / f"{stem}_preprocessed.csv"

    if output_file == Path("-"):
        df.to_csv(sys.stdout.buffer, index=False)
        logger.success("Preprocessed CSV written to stdout.")
    else:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        logger.success(f"Preprocessed CSV saved to {output_file!r}")

    return df


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

    summary = compute_pca_summary(
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
