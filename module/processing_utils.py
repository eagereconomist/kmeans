import pandas as pd
import numpy as np
from pathlib import Path
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
from tennis_racquet_analysis.preprocessing_utils import load_data  # noqa: F401


def write_csv(dataframe: pd.DataFrame, prefix: str, suffix: str, output_dir: Path) -> Path:
    """
    Write `dataframe` to {output_dir}/{prefix}_{suffix}.csv,
    returns the Path to the file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{prefix}_{suffix}.csv"
    dataframe.to_csv(file_path, index=False)
    return file_path


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


def log1p_transform(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = np.log1p(df[num_cols])
    return df


def yeo_johnson(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=np.number).columns
    scaler = preprocessing.PowerTransformer(method="yeo-johnson").fit(df[num_cols])
    df[num_cols] = scaler.transform(df[num_cols])
    return df
