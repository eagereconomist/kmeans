from pathlib import Path
from loguru import logger
from tqdm import tqdm
from typing import List
import typer
from tennis_racquet_analysis.config import RAW_DATA_DIR, INTERIM_DATA_DIR
from tennis_racquet_analysis.preprocessing_utils import (
    load_data,
    check_iqr_outliers,
    drop_column,
    drop_row,
    rename_column,
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
        "preprocessed",
        "--label",
        "-l",
        help="Suffix for the ouput file before .csv",
    ),
    dropped_columns: List[str] = typer.Option(
        [],
        "-dropped-column",
        "-dc",
        help="Name of column to drop; repeat flag to add more.",
    ),
    iqr_check: bool = typer.Option(
        False,
        "--iqr-check",
        "-iqr",
        help="If set, identify IQR outliers in the cleaned DataFrame and print them.",
    ),
    drop_rows: List[int] = typer.Option(
        [], "-dropped-row", "-dr", help="Drop rows by integer index."
    ),
    renamed_columns: List[str] = typer.Option(
        [],
        "--rename-column",
        "-rc",
        help="Name of column to rename (dots -> underscores)"
        "using `rename_column`; repeat flag to add more.",
    ),
):
    input_path = input_dir / input_file
    logger.info(f"Loading raw dataset from: {input_path}...")
    df = load_data(input_path)
    cleaning_steps: list[tuple[str, callable, dict]] = []
    cleaning_steps += [("drop_column", drop_column, {"column": col}) for col in dropped_columns]
    cleaning_steps += [
        ("rename_column", rename_column, {"column": col}) for col in renamed_columns
    ]
    cleaning_steps += [("drop_row", drop_row, {"index_list": [row]}) for row in drop_rows]
    for step_name, func, kwargs in tqdm(
        cleaning_steps, total=len(cleaning_steps), ncols=100, desc="Data Preprocessing Steps"
    ):
        logger.info(f"Applying {step_name}...")
        df = func(df, **kwargs)
    if iqr_check:
        logger.info("Checking for IQR outliers...")
        outliers = check_iqr_outliers(df)
        if outliers.empty:
            logger.info("No IQR-based outliers detected.")
        else:
            typer.echo("\nDetected IQR outliers (row, column, value):")
            typer.echo(outliers.to_frame(name="outlier_value"))
    stem = Path(input_file).stem
    suffix_parts: list[str] = []
    if dropped_columns:
        suffix_parts.append("drop-" + "-".join(dropped_columns))
    if renamed_columns:
        suffix_parts.append("rename-" + "-".join(renamed_columns))
    if drop_rows:
        suffix_parts.append("drop-rows-" + "-".join(str(r) for r in drop_rows))
    suffix = "_".join(suffix_parts)
    output_file = f"{stem}_{file_label}" + (f"_{suffix}" if suffix else "") + ".csv"
    output_path = INTERIM_DATA_DIR / output_file
    df.to_csv(output_path, index=False)
    logger.info(f"Preprocessed DataFrame type: {type(df)}")
    logger.info(f"Preprocessed DataFrame dimensions: {df.shape}")
    logger.success(f"Preprocessed csv saved to {output_path}")
    return df


if __name__ == "__main__":
    app()
