import pandas as pd


def squared(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    dataframe[f"{column}_sq"] = dataframe[column] ** 2
    return dataframe


def apply_interaction(dataframe: pd.DataFrame, column_1: str, column_2: str) -> pd.DataFrame:
    dataframe[f"{column_1}_x_{column_2}"] = dataframe[column_1] * dataframe[column_2]
    return dataframe
