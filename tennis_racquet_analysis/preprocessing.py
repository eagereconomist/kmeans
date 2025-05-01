from pathlib import Path
from loguru import logger
from tqdm import tqdm
from typing import List, Optional
import typer
from tennis_racquet_analysis.config import RAW_DATA_DIR, INTERIM_DATA_DIR
from tennis_racquet_analysis.preprocessing_utils import load_data, drop_column, rename_column

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
    dropped_columns: Optional[List[str]] = typer.Option(
        None,
        "-dropped-column",
        "-dc",
        help="Name of column to drop; repeat flag to add more.",
    ),
    renamed_columns: Optional[List[str]] = typer.Option(
        None,
        "--rename-column",
        "-rc",
        help="Name of column to rename (dots -> underscores)"
        "using `rename_column`; repeat flag to add more.",
    ),
):
    input_path = input_dir / input_file
    logger.info(f"Loading raw dataset from: {input_path}...")
    df = load_data(input_path)
    dropped_cols = dropped_columns or []
    if dropped_cols:
        missing = set(dropped_cols) - set(df.columns)
        if missing:
            raise typer.BadParameter(f"Column(s) not found in DataFrame: {missing!r}")
    renamed_cols = renamed_columns or []
    if renamed_cols:
        missing = set(renamed_cols) - set(df.columns)
        if missing:
            raise typer.BadParameter(f"Column(s) to rename not found: {missing!r}")
    cleaning_steps: list[tuple[str, callable, dict]] = []
    cleaning_steps += [("drop_column", drop_column, {"column": col}) for col in dropped_cols]
    cleaning_steps += [("rename_column", rename_column, {"column": col}) for col in renamed_cols]
    for step_name, func, kwargs in tqdm(
        cleaning_steps, total=len(cleaning_steps), ncols=100, desc="Data Preprocessing Steps:"
    ):
        logger.info(f"Applying {step_name}...")
        df = func(df, **kwargs)

    stem = Path(input_file).stem
    output_file = f"{stem}_{file_label}_{dropped_columns}_{renamed_columns}.csv"
    output_path = INTERIM_DATA_DIR / output_file
    df.to_csv(output_path, index=False)
    logger.info(f"Preprocessed DataFrame type: {type(df)}")
    logger.info(f"Preprocessed DataFrame dimensions: {df.shape}")
    logger.success(f"Preprocessed csv saved to {output_path}")
    return df


if __name__ == "__main__":
    app()
