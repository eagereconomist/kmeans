# from pathlib import Path
import typer
from tqdm import tqdm
from loguru import logger
from tennis_racquet_analysis.config import DATA_DIR
from tennis_racquet_analysis.preprocessing_utils import load_data

app = typer.Typer()


@app.command()
def main(
    dir_label: str,
    input_file: str,
    preview: bool = typer.Option(
        False, "--preview/--no-preview", help="Show first 5 rows of the loaded DataFrame."
    ),
):
    logger.info("Starting Data Ingestion Pipeline")
    pbar = tqdm(total=3, desc="Data Ingestion", ncols=80)
    pbar.set_description("Resolving Input File Path")
    path = DATA_DIR / dir_label / input_file
    pbar.update(1)
    pbar.set_description("Reading File into DataFrame")
    df = load_data(path)
    pbar.update(2)
    pbar.set_description("Finalizing Data Ingestion")
    typer.echo(df.head())
    logger.info(f"Ingested DataFrame Type: {type(df)}")
    logger.info(f"DataFrame Dimensions: {df.shape}")
    pbar.update(3)
    pbar.close()
    logger.success(f"Completed Data Ingestion from {path}")
    return df


if __name__ == "__main__":
    app()
