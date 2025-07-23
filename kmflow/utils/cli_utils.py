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


def comma_split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def comma_split_int(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _write_df(df: pd.DataFrame, output_file: Path) -> None:
    """
    Write df to output_file, or to stdout if output_file == Path('-').
    """
    if output_file == Path("-"):
        df.to_csv(sys.stdout.buffer, index=False)
        logger.success("CSV written to stdout.")
    else:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        logger.success(f"CSV saved to {output_file!r}")


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
