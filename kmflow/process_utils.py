import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from typing import Callable
import numpy as np

from loguru import logger
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler

from cli_utils import read_df

__all__ = [
    "_run_scaler_with_progress",
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


def _run_scaler_with_progress(
    scaler_fn: Callable[[pd.DataFrame], pd.DataFrame],
    input_file: Path,
    output_file: Path,
    suffix: str,
    desc: str,
) -> None:
    """
    Read via read_df, apply scaler_fn, then write via write_csv—
    all with a 3-step tqdm progress bar (read, transform, write).
    If output_file == Path('-'), write the final CSV to stdout.
    """
    with tqdm(total=3, desc=desc, colour="green") as pbar:
        # 1) Read
        df = read_df(input_file)
        pbar.update(1)

        # 2) Transform
        df_out = scaler_fn(df)
        pbar.update(1)

        # 3) Write
        if output_file == Path("-"):
            df_out.to_csv(sys.stdout.buffer, index=False)
            logger.success("CSV written to stdout.")
        else:
            write_csv(
                df_out,
                prefix=input_file.stem,
                suffix=suffix,
                output_dir=output_file,
            )
        pbar.update(1)


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
