from pathlib import Path
import pandas as pd
from typing import Callable
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler

from cli_utils import read_df, _write_df

__all__ = [
    "_run_scaler",
    "apply_normalizer",
    "apply_standardization",
    "apply_minmax",
    "apply_log1p",
    "apply_yeo_johnson",
]


def write_csv(dataframe: pd.DataFrame, prefix: str, suffix: str, output_dir: Path) -> Path:
    """
    Write `dataframe` to {output_dir}/{prefix}_{suffix}.csv,
    returns the Path to the file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{prefix}_{suffix}.csv"
    dataframe.to_csv(file_path, index=False)
    return file_path


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
