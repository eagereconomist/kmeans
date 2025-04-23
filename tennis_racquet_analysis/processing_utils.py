import pandas as pd
import numpy as np
from pathlib import Path
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
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


def apply_normalizer(dataframe):
    scaler = Normalizer()
    scaled = scaler.fit_transform(dataframe)
    return pd.DataFrame(scaled, columns=dataframe.columns)


def apply_standardization(dataframe):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(dataframe)
    return pd.DataFrame(scaled, columns=dataframe.columns)


def apply_minmax(dataframe):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(dataframe)
    return pd.DataFrame(scaled, columns=dataframe.columns)


def log_transform(dataframe):
    scaler = FunctionTransformer(np.log1p, validate=True)
    scaled = scaler.fit_transform(dataframe)
    return pd.DataFrame(scaled, columns=dataframe.columns)


def yeo_johnson(dataframe):
    scaler = preprocessing.PowerTransformer(method="yeo-johnson", standardize=True)
    scaled = scaler.fit_transform(dataframe)
    return pd.DataFrame(scaled, columns=dataframe.columns)
