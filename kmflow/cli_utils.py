import sys
import pandas as pd
from pathlib import Path
from loguru import logger


def read_df(path: Path) -> pd.DataFrame:
    """Read CSV from file or stdin ('-')."""
    if path == Path("-"):
        logger.info("Reading DataFrame from stdin...")
        return pd.read_csv(sys.stdin)
    logger.info(f"Reading DataFrame from {path!r}...")
    df = pd.read_csv(path)
    logger.info("Data loaded.")
    return df
