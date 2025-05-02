from pathlib import Path
from loguru import logger
from tqdm import tqdm
from typing import List
from itertools import combinations
import typer
from tennis_racquet_analysis.config import INTERIM_DATA_DIR
from tennis_racquet_analysis.preprocessing_utils import load_data
from tennis_racquet_analysis.features_utils import squared, apply_interaction

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
    output_dir: Path = typer.Option(
        INTERIM_DATA_DIR,
        "--output-dir",
        "-o",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Directory to write feature-engineered csv to.",
    ),
    file_label: str = typer.Option(
        "features",
        "--label",
        "-l",
        help="Suffix for the output file before `.csv`.",
    ),
    squared_columns: List[str] = typer.Option(
        [],
        "--squared-column",
        "-sc",
        help="Name of column to square; repeat flag to add more.",
    ),
    interaction_columns: List[str] = typer.Option(
        [],
        "--interaction-column",
        "-ic",
        help="Name of columns to multiply; repeat flag to add more.",
    ),
):
    """
    Load a preprocessed csv, square each of the given columns,
    and write out a new feature-engineered csv.
    """
    input_path = input_dir / input_file
    logger.info(f"Loading preprocessed dataset from {input_path!r}")
    df = load_data(input_path)
    feature_steps = [
        (f"squared_columns {col}", squared, {"column": col}) for col in squared_columns
    ]
    for col_1, col_2 in combinations(interaction_columns, 2):
        feature_steps.append(
            (
                f"interact_{col_1}_x_{col_2}",
                apply_interaction,
                {"column_1": col_1, "column_2": col_2},
            )
        )
    for step_name, func, kwargs in tqdm(
        feature_steps, total=len(feature_steps), ncols=100, desc="Feature Engineering Steps"
    ):
        logger.info(f"Applying {step_name}...")
        df = func(df, **kwargs)

    stem = Path(input_file).stem
    output_file = f"{stem}_{file_label}.csv"
    output_path = output_dir / output_file
    df.to_csv(output_path, index=False)
    logger.info(f"Feature-engineered DataFrame type: {type(df)}")
    logger.info(f"Feature-engineered DataFrame dimensions: {df.shape}")
    logger.success(f"Feature-engineered DataFrame saved to {output_path!r}")
    return df


if __name__ == "__main__":
    app()
