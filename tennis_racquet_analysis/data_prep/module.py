import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
from sklearn.preprocessing import FunctionTransformer


def load_data():
    data_path = os.path.join("data", "raw", "tennis_racquets.csv")
    print("Looking for file at:", data_path)
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print("Data loaded successfully!")
        print(df.head(10))
        return df
    else:
        raise FileNotFoundError(f"File not found. Please check your path: {data_path}")


def drop_column(dataframe, column):
    return dataframe.drop(columns=[column])


def rename_column(dataframe, column):
    new_column = column.replace(".", "")
    return dataframe.rename(columns={column: new_column})


def squared(dataframe, column):
    dataframe[f"{column}_sq"] = dataframe[column] ** 2
    return dataframe


def write_csv(dataframe, subfolder, file_label):
    file_path = os.path.join("data", subfolder, f"tennis_racquet_{file_label}.csv")
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
