import sys
from pathlib import Path
import pandas as pd
from loguru import logger


# ─── Send all loguru output to stderr ─────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - {message}",
    colorize=True,
)


def read_df(path: Path) -> pd.DataFrame:
    """Read CSV from file or stdin ('-').
    Logs to stderr so stdout stays clean
    for data."""
    if path == Path("-"):
        logger.info("Reading DataFrame from stdin...")
        return pd.read_csv(sys.stdin)
    logger.info(f"Reading DataFrame from {path!r}...")
    df = pd.read_csv(path)
    logger.info("Data loaded.")
    return df
