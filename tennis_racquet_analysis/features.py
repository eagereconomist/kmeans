from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
from tennis_racquet_analysis.config import INTERIM_DATA_DIR
from tennis_racquet_analysis.preprocessing_utils import load_data, squared

app = typer.Typer()


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "tennis_racquets_preprocessed.csv",
    file_label: str = "features",
):
    output_path: Path = INTERIM_DATA_DIR / f"tennis_racquets_{file_label}.csv"
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

    df.to_csv(output_path, index=False)
    logger.success(f"Feature-engineered dataset saved to {output_path}")


if __name__ == "__main__":
    app()
