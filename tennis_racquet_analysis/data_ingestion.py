import typer
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from tennis_racquet_analysis.config import RAW_DATA_DIR
from tennis_racquet_analysis.preprocessing_utils import load_data

app = typer.Typer()


@app.command()
def main(
    input_file: str,
    preview: bool = typer.Option(
        False, "--preview/--no-preview", help="Show first 5 rows of the loaded DataFrame."
    ),
):
    logger.info("Starting data ingestion pipeline...")
    pbar = tqdm(total=1, desc="Data Ingestion", ncols=100)
    input_path: Path = RAW_DATA_DIR / input_file
    df = load_data(input_path)
    pbar.set_description("Finalizing Data Ingestion")
    pbar.update(1)
    pbar.close()
    # Only show the first 5 rows of DataFrame if --preview is set
    if preview:
        typer.echo(df.head())
    logger.info(f"Ingested DataFrame type: {type(df)}")
    logger.info(f"DataFrame dimensions: {df.shape}")
    logger.success(f"Completed data ingestion from {input_path!r}")
    return df


if __name__ == "__main__":
    app()
