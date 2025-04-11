from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd

from tennis_racquet_analysis.config import RAW_DATA_DIR, INTERIM_DATA_DIR
from tennis_racquet_analysis.preprocessing_utils import load_data, drop_column, rename_column

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "tennis_racquets.csv", file_label: str = "preprocessed"
):
    output_path: Path = INTERIM_DATA_DIR / f"tennis_racquets_{file_label}.csv"
    logger.info("Loading raw dataset...")
    df = load_data(input_path)
    cleaning_steps = [
        ("drop_column", drop_column, {"column": "Racquet"}),
        ("rename_column", rename_column, {"column": "static.weight"}),
    ]

    for step_name, func, kwargs in tqdm(
        cleaning_steps, total=len(cleaning_steps), desc="Data Preprocessing Steps:"
    ):
        logger.info(f"Applying {step_name}...")
        df = func(df, **kwargs)

    df.to_csv(output_path, index=False)
    logger.success(f"Preprocessed dataset saved to {output_path}")


if __name__ == "__main__":
    app()
