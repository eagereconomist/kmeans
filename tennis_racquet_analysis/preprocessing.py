from pathlib import Path
from loguru import logger
from tqdm import tqdm
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
):
    input_path = input_dir / input_file
    logger.info(f"Loading raw dataset from: {input_path}...")
    df = load_data(input_path)
    cleaning_steps = [
        ("drop_column", drop_column, {"column": "Racquet"}),
        ("rename_column", rename_column, {"column": "static.weight"}),
    ]

    for step_name, func, kwargs in tqdm(
        cleaning_steps, total=len(cleaning_steps), ncols=100, desc="Data Preprocessing Steps:"
    ):
        logger.info(f"Applying {step_name}...")
        df = func(df, **kwargs)

    stem = Path(input_file).stem
    output_file = f"{stem}_{file_label}.csv"
    output_path = INTERIM_DATA_DIR / output_file
    df.to_csv(output_path, index=False)
    logger.success(f"Preprocessed csv saved to {output_path}")


if __name__ == "__main__":
    app()
