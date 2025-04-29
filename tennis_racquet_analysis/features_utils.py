import pandas as pd


def squared(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    dataframe[f"{column}_sq"] = dataframe[column] ** 2
    return dataframe
