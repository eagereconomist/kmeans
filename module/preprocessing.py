from pathlib import Path
from loguru import logger
from tqdm import tqdm
from typing import List, Optional
import typer
from tennis_racquet_analysis.config import (
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    DATA_DIR,
)
from tennis_racquet_analysis.preprocessing_utils import (
    load_data,
    find_iqr_outliers,
    compute_pca_summary,
    drop_column,
    drop_row,
    dotless_column,
)
from tennis_racquet_analysis.processing_utils import (
    write_csv,
)

app = typer.Typer()


@app.command()
def main(
    input_file: str = typer.Argument(..., help="Raw csv filename."),
    input_dir: Path = typer.Option(
        RAW_DATA_DIR,
        "--input-dir",
        "-d",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Directory where raw files live.",
    ),
    file_label: str = typer.Option(
        None,
        "--label",
        "-l",
        help="Suffix for the ouput file before .csv",
    ),
    dropped_columns: List[str] = typer.Option(
        [],
        "--dropped-column",
        "-dc",
        help="Name of column to drop; repeat flag to add more.",
    ),
    iqr_check: bool = typer.Option(
        False,
        "--iqr-check",
        "-iqr",
        help="If set, identify IQR outliers in the cleaned DataFrame and print them.",
    ),
    export_outliers: bool = typer.Option(
        False,
        "--export-outliers",
        "-eo",
        help="If set, write outliers to the default data/interim/iqr_outliers.csv",
    ),
    remove_outliers: bool = typer.Option(
        False,
        "--remove-outliers",
        "-ro",
        help="When set, drop all rows containing outliers from the working `df`.",
    ),
    drop_rows: List[int] = typer.Option(
        [], "--dropped-row", "-dr", help="Drop rows by integer index."
    ),
    dotless_columns: List[str] = typer.Option(
        [],
        "--dotless-column",
        "-dot",
        help="Name of column to switch out dot for empty string"
        "using `dotless_column`; repeat flag to add more.",
    ),
):
    input_path = input_dir / input_file
    logger.info(f"Loading raw dataset from: {input_path}...")
    df = load_data(input_path)
    cleaning_steps: list[tuple[str, callable, dict]] = []
    cleaning_steps += [("drop_column", drop_column, {"column": col}) for col in dropped_columns]
    cleaning_steps += [
        ("dotless_column", dotless_column, {"column": col}) for col in dotless_columns
    ]
    cleaning_steps += [("drop_row", drop_row, {"index_list": [row]}) for row in drop_rows]
    for step_name, func, kwargs in tqdm(
        cleaning_steps, total=len(cleaning_steps), ncols=100, desc="Data Preprocessing Steps"
    ):
        logger.info(f"Applying {step_name}...")
        df = func(df, **kwargs)
    if iqr_check:
        logger.info("Finding IQR outliers…")
        outlier_series = find_iqr_outliers(df)

        if outlier_series.empty:
            logger.info("No IQR‐based outliers detected.")
        else:
            outlier_df = outlier_series.reset_index().rename(
                columns={"level_0": "row_index", "level_1": "column", 0: "outlier_value"}
            )
            if export_outliers:
                write_csv(outlier_df, prefix="iqr", suffix="outliers", output_dir=INTERIM_DATA_DIR)
                logger.success(f"Outliers written to {INTERIM_DATA_DIR / 'iqr_outliers.csv'!r}")
            else:
                typer.echo("\nDetected IQR outliers:")
                typer.echo(outlier_df)
            if remove_outliers:
                rows_to_drop = outlier_df["row_index"].unique().tolist()
                total_rows = df.shape[0]
                df = drop_row(df, rows_to_drop)
                rows_after_outliers_dropped = df.shape[0]
                logger.success(f"Removed {len(rows_to_drop)} outlier rows: {rows_to_drop}")
                logger.success(
                    f"DataFrame went from {total_rows} to {rows_after_outliers_dropped} rows"
                )
    stem = Path(input_file).stem
    if file_label:
        output_filename = f"{stem}_{file_label}.csv"
    else:
        suffix_parts: list[str] = []
        if dropped_columns:
            suffix_parts.append("drop-" + "_".join(dropped_columns))
        if dotless_columns:
            suffix_parts.append("dotless-" + "_".join(dotless_columns))
        if drop_rows:
            suffix_parts.append("drop-rows-" + "_".join(map(str, drop_rows)))
        base_label = "preprocessed"
        if suffix_parts:
            base_label += "_" + "_".join(suffix_parts)
        output_filename = f"{stem}_{suffix_parts}.csv"
    output_path = INTERIM_DATA_DIR / output_filename
    df.to_csv(output_path, index=False)
    logger.info(f"Preprocessed DataFrame type: {type(df)}")
    logger.info(f"Preprocessed DataFrame dimensions: {df.shape}")
    logger.success(f"Preprocessed CSV saved to {output_path!r}")
    return df


@app.command("pca-summary")
def pca_summary(
    input_file: str = typer.Argument(..., help="csv filename under data subfolder."),
    input_dir: Path = typer.Option(
        "processed",
        "--input-dir",
        "-d",
        help="Sub-folder under data/ (e.g. external, interim, processed, raw), where the input file lives.",
    ),
    feature_columns: list[str] = typer.Option(
        None,
        "--feature-column",
        "-f",
        help="Name of numeric column to include; repeat flag to add more. Defaults to all numeric.",
    ),
    n_components: Optional[int] = typer.Option(
        None,
        "--n-components",
        "-n",
        help="How many principal components to compute/export (defaults to all).",
    ),
    random_state: int = typer.Option(
        4572, "--seed", "-s", help="Random seed for reproducibility."
    ),
    output_dir: Path = typer.Option(
        "processed",
        "--output-dir",
        "-o",
        help="Sub-folder under data/ (e.g. external, interim, processed, raw), where the input file lives.",
    ),
):
    input_path = DATA_DIR / input_dir / input_file
    output_path = DATA_DIR / output_dir
    df = load_data(input_path)
    dict_pca = compute_pca_summary(
        df=df,
        feature_columns=feature_columns,
        n_components=None,
        random_state=random_state,
    )
    stem = Path(input_file).stem
    df_loadings = dict_pca["loadings"]
    df_loadings = df_loadings.reset_index().rename(columns={"index": "component"})
    loadings_path = write_csv(
        df_loadings, prefix=stem, suffix="pca_loadings", output_dir=output_path
    )
    logger.success(f"Saved PCA Loadings → {loadings_path!r}")

    df_scores_full = dict_pca["scores"]
    if n_components is not None:
        df_scores = df_scores_full.iloc[:, :n_components].reset_index(drop=True)
    else:
        df_scores = df_scores_full.reset_index(drop=True)
    suffix = f"pca_scores_{df_scores.shape[1]}pc" if n_components else "pca_scores"
    scores_path = write_csv(
        df_scores,
        prefix=stem,
        suffix=suffix,
        output_dir=output_path,
    )
    logger.success(f"Saved PCA Scores → {scores_path!r}")

    df_pve = dict_pca["pve"]
    df_pve = df_pve.reset_index().rename(columns={"index": "component"})
    pve_path = write_csv(
        df_pve,
        prefix=stem,
        suffix="pca_proportion_var",
        output_dir=output_path,
    )
    logger.success(f"Saved Explained Variance Ratio → {pve_path!r}")

    df_cpve = dict_pca["cpve"]
    df_cpve = df_cpve.reset_index().rename(columns={"index": "component"})
    cpve_path = write_csv(
        df_cpve,
        prefix=stem,
        suffix="pca_cumulative_var",
        output_dir=output_path,
    )
    logger.success(f"Saved Cumulative Variance Ratio → {cpve_path!r}")


if __name__ == "__main__":
    app()
