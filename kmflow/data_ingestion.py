import typer
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from kmflow.config import RAW_DATA_DIR
from kmflow.preprocessing_utils import load_data

app = typer.Typer()


@app.command()
def main(
    input_file: str = typer.Argument(..., help="Raw data csv."),
    preview: bool = typer.Option(
        False, "--preview", "-p", help="Show first 5 rows of the loaded DataFrame."
    ),
):
    logger.info("Starting data ingestion pipeline...")
    pbar = tqdm(total=1, desc="Data Ingestion", ncols=100)
    input_path: Path = RAW_DATA_DIR / input_file
    df = load_data(input_path)
    pbar.set_description("Finalizing Data Ingestion")
    pbar.update(1)
    pbar.close()
    if preview:
        typer.echo(df.head())
    logger.info(f"Ingested DataFrame type: {type(df)}")
    logger.info(f"DataFrame dimensions: {df.shape}")
    logger.success(f"Completed data ingestion from {input_path!r}")
    return df


if __name__ == "__main__":
    app()
