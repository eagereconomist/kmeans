from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import os

from tennis_racquet_analysis.config import INTERIM_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def load_data(input_path: Path) -> pd.DataFrame:
    """
    Loads the raw tennis racquet data from the provided input path.
    """
    logger.info(f"Looking for file at: {input_path}")
    if input_path.exists():
        df = pd.read_csv(input_path)
        logger.info("Data loaded successfully!")
        return df
    else:
        raise FileNotFoundError(f"File not found. Please check your path: {input_path}")


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "tennis_racquets.csv", file_label: str = "preprocessed"
):
    """
    Processes the dataset by loading it from the input path and saving it to a dynamically generated output path.
    """
    # Construct the output path based on the file_label parameter
    output_path: Path = INTERIM_DATA_DIR / f"tennis_racquets_{file_label}.csv"

    logger.info("Processing dataset...")

    # Use the parameterized load_data function to load data from input_path
    df = load_data(input_path)

    # Simulate processing steps; replace this block with actual cleaning/transformation logic.
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")

    # Save the (processed) DataFrame to the output path.
    df.to_csv(output_path, index=False)
    logger.success(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    app()
