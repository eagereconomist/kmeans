import pandas as pd


def squared(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df[f"{column}_sq"] = df[column] ** 2
    return df


def apply_interaction(df: pd.DataFrame, column_1: str, column_2: str) -> pd.DataFrame:
    df[f"{column_1}_x_{column_2}"] = df[column_1] * df[column_2]
    return df
