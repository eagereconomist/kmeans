from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
from tennis_racquet_analysis.config import RAW_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()


def load_data(input_path: Path) -> pd.DataFrame:
    logger.info(f"Looking for file at: {input_path}")
    if input_path.exists():
        df = pd.read_csv(input_path)
        logger.info("Data loaded successfully!")
        return df
    else:
        raise FileNotFoundError(f"File not found. Please check your path: {input_path}")


def drop_column(dataframe, column):
    return dataframe.drop(columns=[column])


def rename_column(dataframe, column):
    new_column = column.replace(".", "")
    return dataframe.rename(columns={column: new_column})


def squared(dataframe, column):
    dataframe[f"{column}_sq"] = dataframe[column] ** 2
    return dataframe


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "tennis_racquets.csv", file_label: str = "preprocessed"
):
    output_path: Path = INTERIM_DATA_DIR / f"tennis_racquets_{file_label}.csv"
    logger.info("Processing dataset...")
    df = load_data(input_path)

    # Each tuple consists of: (Step name, Function, **kwargs as a dictionary)
    cleaning_steps = [
        ("drop_column", drop_column, {"column": "Racquet"}),
        ("rename_column", rename_column, {"column": "static.weight"}),
        ("squared headsize", squared, {"column": "headsize"}),
        ("squared swingweight", squared, {"column": "swingweight"}),
    ]

    # Iterate over each cleaning step with a progress bar.
    for step_name, func, kwargs in tqdm(
        cleaning_steps, total=len(cleaning_steps), desc="Data Preprocessing Steps:"
    ):
        logger.info(f"Applying {step_name}...")
        df = func(df, **kwargs)  # Apply the function with its parameters and update the DataFrame

    # After all cleaning steps are applied, save the processed dataset.
    df.to_csv(output_path, index=False)
    logger.success(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    app()
