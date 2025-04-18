from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
from tennis_racquet_analysis.config import INTERIM_DATA_DIR
from tennis_racquet_analysis.preprocessing_utils import load_data
from tennis_racquet_analysis.features_utils import squared

app = typer.Typer()


@app.command()
def main(
    input_file: str = typer.Argument(..., help="Preprocessed csv filename."),
    input_dir: Path = typer.Option(
        INTERIM_DATA_DIR,
        "--input-dir",
        "-d",
        exists=True,
        file_okay=True,
        help="Directory where preprocessed files live.",
    ),
    file_label: str = typer.Option(
        "features",
        "--label",
        "-l",
        help="Suffix for the output file before .csv",
    ),
):
    input_path = input_dir / input_file
    logger.info("Loading preprocessed dataset...")
    df = load_data(input_path)
    feature_steps = [
        ("squared headsize", squared, {"column": "headsize"}),
        ("squared swingweight", squared, {"column": "swingweight"}),
    ]

    for step_name, func, kwargs in tqdm(
        feature_steps, total=len(feature_steps), desc="Feature Engineering Steps"
    ):
        logger.info(f"Applying {step_name}...")
        df = func(df, **kwargs)

    stem = Path(input_file).stem
    output_file = f"{stem}_{file_label}.csv"
    output_path = INTERIM_DATA_DIR / output_file
    df.to_csv(output_path, index=False)
    logger.success(f"Feature-engineered dataset saved to {output_path}")


if __name__ == "__main__":
    app()
