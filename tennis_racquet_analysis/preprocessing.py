from pathlib import Path
from loguru import logger
from tqdm import tqdm
from typing import List
import typer
from tennis_racquet_analysis.config import RAW_DATA_DIR, INTERIM_DATA_DIR
from tennis_racquet_analysis.preprocessing_utils import (
    load_data,
    find_iqr_outliers,
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
                df = drop_row(df, rows_to_drop)
                logger.success(f"Removed outlier rows: {rows_to_drop}")
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


if __name__ == "__main__":
    app()
