#!/usr/bin/env python
import sys
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer

from kmflow.preprocess_utils import (
    find_iqr_outliers,
    compute_pca_summary,
    drop_column,
    drop_row,
    dotless_column,
)
from kmflow.process_utils import write_csv
from kmflow.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer(help="Data preprocessing and PCA-summary commands.")


@app.command("preprocess")
def preprocess(
    input_file: Path = typer.Argument(
        ...,
        help="Raw CSV file to read (use '-' to read from stdin).",
    ),
    dropped_columns: list[str] = typer.Option(
        [],
        "--dropped-column",
        "-dc",
        help="Name of column to drop; can be repeated.",
    ),
    dotless_columns: list[str] = typer.Option(
        [],
        "--dotless-column",
        "-dot",
        help="Name of column whose dots to remove; can be repeated.",
    ),
    drop_rows: list[int] = typer.Option(
        [],
        "--dropped-row",
        "-dr",
        help="Row indices to drop; can be repeated.",
    ),
    iqr_check: bool = typer.Option(
        False,
        "--iqr-check",
        "-iqr",
        help="Identify IQR-based outliers and report them.",
    ),
    export_outliers: bool = typer.Option(
        False,
        "--export-outliers",
        "-eo",
        help="Write detected outliers to interim/iqr_outliers.csv",
    ),
    remove_outliers: bool = typer.Option(
        False,
        "--remove-outliers",
        "-ro",
        help="Drop all rows containing IQR outliers.",
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
        help="Where to save the preprocessed CSV; use '-' for stdout.",
    ),
):
    """
    Apply column/row drops, dotless renaming, and optional IQR outlier handling
    to a raw CSV. Outputs a cleaned CSV.
    """
    # ─── 1) read ───────────────────────────────────────────────────────
    if input_file == Path("-"):
        df = pd.read_csv(sys.stdin)
    else:
        df = pd.read_csv(input_file)

    # ─── 2) build transform steps ────────────────────────────────────
    steps: list[tuple[str, callable, list]] = []
    for col in dropped_columns:
        steps.append(("drop_column", drop_column, [col]))
    for col in dotless_columns:
        steps.append(("dotless_column", dotless_column, [col]))
    for idx in drop_rows:
        steps.append(("drop_row", drop_row, [[idx]]))

    # ─── 3) apply them ────────────────────────────────────────────────
    for name, func, args in tqdm(steps, desc="Data Preprocessing Steps", ncols=100):
        logger.info(f"Applying {name}...")
        df = func(df, *args)

    # ─── 4) optional IQR outlier reporting / removal ─────────────────
    if iqr_check:
        logger.info("Finding IQR outliers…")
        out = find_iqr_outliers(df)

        if out.empty:
            logger.info("No IQR-based outliers detected.")
        else:
            out_df = out.reset_index().rename(
                columns={"level_0": "row_index", "level_1": "column", 0: "outlier_value"}
            )
            if export_outliers:
                # always write to fixed filename
                write_csv(out_df, prefix="iqr", suffix="outliers", output_dir=INTERIM_DATA_DIR)
                logger.success(f"Outliers written to {INTERIM_DATA_DIR / 'iqr_outliers.csv'!r}")
            else:
                typer.echo("\nDetected IQR outliers:")
                typer.echo(out_df)

            if remove_outliers:
                rows = out_df["row_index"].unique().tolist()
                before = df.shape[0]
                df = drop_row(df, rows)
                after = df.shape[0]
                logger.success(f"Removed {len(rows)} outlier rows: {rows}")
                logger.success(f"Row count: {before} -> {after}")

    # ─── 5) preview if requested ─────────────────────────────────────
    if preview:
        typer.echo(df.head())

    # ─── 6) write out ────────────────────────────────────────────────
    if output_file is None:
        output_file = INTERIM_DATA_DIR / "preprocessed.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file == Path("-"):
        df.to_csv(sys.stdout, index=False)
        logger.success("Preprocessed CSV written to stdout.")
    else:
        df.to_csv(output_file, index=False)
        logger.success(f"Preprocessed CSV saved to {output_file!r}")

    return df


@app.command("pca")
def pca_summary(
    input_file: Path = typer.Argument(
        ...,
        help="CSV file to read (use '-' to read from stdin).",
    ),
    feature_columns: list[str] = typer.Option(
        None,
        "--numeric-cols",
        "-numeric-cols",
        help="Numeric column to include; can be repeated. Defaults to all numeric.",
    ),
    n_components: int = typer.Option(
        None,
        "--n-components",
        "-n",
        help="Number of PCs to compute (defaults to all).",
    ),
    random_state: int = typer.Option(
        4572, "--seed", "-s", help="Random seed for reproducibility."
    ),
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR,
        "--output-dir",
        "-o",
        dir_okay=True,
        file_okay=False,
        help="Where to save the PCA summary CSVs.",
    ),
):
    """
    Compute PCA loadings, scores, explained variance, and cumulative variance,
    then write four CSVs:
      1) pca_loadings.csv
      2) pca_scores_<k>pc.csv
      3) pca_proportion_var.csv
      4) pca_cumulative_var.csv
    """
    # ─── read ─────────────────────────────────────────────────────────
    if input_file == Path("-"):
        df = pd.read_csv(sys.stdin)
    else:
        df = pd.read_csv(input_file)

    # ─── compute ───────────────────────────────────────────────────────
    summary = compute_pca_summary(
        df=df,
        feature_columns=feature_columns,
        n_components=n_components,
        random_state=random_state,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── write loadings ───────────────────────────────────────────────
    ld = summary["loadings"].reset_index().rename(columns={"index": "component"})
    p1 = write_csv(ld, suffix="pca_loadings", output_dir=output_dir)
    logger.success(f"Saved PCA Loadings -> {p1!r}")

    # ─── write scores ────────────────────────────────────────────────
    sc = summary["scores"]
    suffix2 = f"pca_scores_{sc.shape[1]}pc" if n_components else "pca_scores"
    p2 = write_csv(sc.reset_index(drop=True), suffix=suffix2, output_dir=output_dir)
    logger.success(f"Saved PCA Scores -> {p2!r}")

    # ─── write explained var ──────────────────────────────────────────
    pv = summary["pve"].reset_index().rename(columns={"index": "component"})
    p3 = write_csv(pv, suffix="pca_proportion_var", output_dir=output_dir)
    logger.success(f"Saved Explained Variance -> {p3!r}")

    # ─── write cumulative var ─────────────────────────────────────────
    cv = summary["cpve"].reset_index().rename(columns={"index": "component"})
    p4 = write_csv(cv, suffix="pca_cumulative_var", output_dir=output_dir)
    logger.success(f"Saved Cumulative Variance -> {p4!r}")


if __name__ == "__main__":
    app()
