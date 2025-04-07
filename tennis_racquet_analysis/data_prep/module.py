import os
import pandas as pd
import numpy as np
from sklearn import preprocessing


def load_data():
    """Load the csv file and return a DataFrame"""
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
