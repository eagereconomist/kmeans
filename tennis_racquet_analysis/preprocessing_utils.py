from pathlib import Path
import pandas as pd
from loguru import logger
from typing import Optional, Sequence, Iterable
from sklearn.decomposition import PCA


def load_data(input_path: Path) -> pd.DataFrame:
    logger.info(f"Looking for file at: {input_path}")
    if input_path.exists():
        df = pd.read_csv(input_path)
        logger.info("Data loaded successfully!")
        return df
    else:
        raise FileNotFoundError(f"File not found. Please check your path: {input_path}")


def find_iqr_outliers(df: pd.DataFrame) -> pd.Series:
    num_df = df.select_dtypes(include="number")
    q1 = num_df.quantile(0.25)
    q3 = num_df.quantile(0.75)
    iqr = q3 - q1
    lower_lim = q1 - 1.5 * iqr
    upper_lim = q3 + 1.5 * iqr
    outlier_mask = (num_df < lower_lim) | (num_df > upper_lim)
    iqr_outliers = num_df.where(outlier_mask).stack()
    return iqr_outliers


def compute_pca_components(
    df: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
    n_components: Optional[int] = None,
    random_state: int = 4572,
) -> pd.DataFrame:
    if feature_columns is None:
        feature_columns = df.select_dtypes(include="number").columns.tolist()
    X = df[feature_columns].values
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X)
    components = pca.components_
    pc_names = [f"PC{i + 1}" for i in range(components.shape[0])]
    return pd.DataFrame(components, index=pc_names, columns=feature_columns)


def drop_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return df.drop(columns=[column])


def drop_row(df: pd.DataFrame, index_list: list[int]) -> pd.DataFrame:
    return df.drop(index=index_list).reset_index(drop=True)


def dotless_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    new_column = column.replace(".", "")
    return df.rename(columns={column: new_column})
