from pathlib import Path
from loguru import logger
from tqdm import tqdm
from typing import List
import typer
from tennis_racquet_analysis.config import INTERIM_DATA_DIR
from tennis_racquet_analysis.preprocessing_utils import load_data
from tennis_racquet_analysis.features_utils import squared

app = typer.Typer()


@app.command()
def main(
    input_file: str = typer.Argument(..., help="Preprocessed csv filename from (data/interim)."),
    input_dir: Path = typer.Option(
        INTERIM_DATA_DIR,
        "--input-dir",
        "-d",
        exists=True,
        file_okay=True,
        dir_okay=True,
        help="Directory where preprocessed files live.",
    ),
    file_label: str = typer.Option(
        "features",
        "--label",
        "-l",
        help="Suffix for the output file before `.csv`.",
    ),
    columns: List[str] = typer.Option(
        ["headsize", "swingweight"],
        "--column",
        "-c",
        help="Name of column to square; repeat flag to add more.",
        show_default=True,
    ),
):
    """
    Load a preprocessed csv, square each of the given columns,
    and write out a new feature-engineered csv.
    """
    input_path = input_dir / input_file
    logger.info(f"Loading preprocessed dataset from {input_path!r}")
    df = load_data(input_path)
    feature_steps = [(f"squared {col}", squared, {"column": col}) for col in columns]

    for step_name, func, kwargs in tqdm(
        feature_steps, total=len(feature_steps), ncols=100, desc="Feature Engineering Steps"
    ):
        logger.info(f"Applying {step_name}...")
        df = func(df, **kwargs)

    stem = Path(input_file).stem
    output_file = f"{stem}_{file_label}.csv"
    output_path = INTERIM_DATA_DIR / output_file
    df.to_csv(output_path, index=False)
    logger.info(f"Feature-engineered DataFrame type: {type(df)}")
    logger.info(f"Feature-engineered DataFrame dimensions: {df.shape}")
    logger.success(f"Feature-engineered DataFrame saved to {output_path!r}")
    return df


if __name__ == "__main__":
    app()
