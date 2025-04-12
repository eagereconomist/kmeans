from pathlib import Path
import pandas as pd
from loguru import logger


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
