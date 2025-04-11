from pathlib import Path
from loguru import logger
import typer
from tennis_racquet_analysis.config import RAW_DATA_DIR
from tennis_racquet_analysis.preprocessing_utils import load_data

app = typer.Typer()


@app.command()
def main(input_path: Path = RAW_DATA_DIR / "tennis_racquets.csv"):
    logger.info("Loading raw dataset...")
    df = load_data(input_path)
    logger.info(f"Type of loaded data: {type(df)}")
    logger.info(f"Data shape: {df.shape}")
    typer.echo(df.head())
    logger.success("Dataset successfully loaded!")
    return df


if __name__ == "__main__":
    app()
