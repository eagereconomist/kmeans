import sys
from pathlib import Path
from loguru import logger
import pandas as pd
from typing import Callable
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler

from cli_utils import read_df


def _write_df(df: pd.DataFrame, output_file: Path) -> None:
    """Write df to output_file or stdout if '-' is given."""
    if output_file == Path("-"):
        df.to_csv(sys.stdout.buffer, index=False)
        logger.success("CSV written to stdout.")
    else:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        logger.success(f"CSV saved to {output_file!r}")


def _run_scaler(
    scaler_fn: Callable[[pd.DataFrame], pd.DataFrame],
    input_file: Path,
    output_file: Path | None,
    suffix: str,
) -> None:
    """
    Read with read_df, apply scaler_fn, write with _write_df.
    If output_file is None, defaults to cwd / f"{stem}_{suffix}.csv"."""
    df = read_df(input_file)
    df_out = scaler_fn(df.copy())
    if output_file is None:
        stem = input_file.stem if input_file != Path("-") else "stdin"
        output_file = Path.cwd() / f"{stem}_{suffix}.csv"
    _write_df(df_out, output_file)


def apply_normalizer(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=np.number).columns
    scaler = Normalizer().fit(df[num_cols])
    df[num_cols] = scaler.transform(df[num_cols])
    return df


def apply_standardization(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=np.number).columns
    scaler = StandardScaler().fit(df[num_cols])
    df[num_cols] = scaler.transform(df[num_cols])
    return df


def apply_minmax(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=np.number).columns
    scaler = MinMaxScaler().fit(df[num_cols])
    df[num_cols] = scaler.transform(df[num_cols])
    return df


def apply_log1p(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = np.log1p(df[num_cols])
    return df


def apply_yeo_johnson(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=np.number).columns
    scaler = preprocessing.PowerTransformer(method="yeo-johnson").fit(df[num_cols])
    df[num_cols] = scaler.transform(df[num_cols])
    return df
