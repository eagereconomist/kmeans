import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from tennis_racquet_analysis.config import PROCESSED_DATA_DIR


def write_csv(dataframe, subfolder, file_label):
    file_path = PROCESSED_DATA_DIR / f"tennis_racquets_{file_label}.csv"
    dataframe.to_csv(file_path, index=False)
    print(f"csv written to {file_path}")
    return dataframe


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
