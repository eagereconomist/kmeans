from pathlib import Path
import pandas as pd
from loguru import logger
from typing import Optional, Sequence, Dict, Union
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


def compute_pca_summary(
    df: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
    hue_column: Optional[str] = None,
    n_components: Optional[int] = None,
    random_state: int = 4572,
) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
    if feature_columns is None:
        all_numueric_cols = df.select_dtypes(include="number").columns.tolist()
        feature_columns = [c for c in all_numueric_cols if c != hue_column]
    X = df[feature_columns].values
    pca = PCA(n_components=n_components, random_state=random_state).fit(X)
    scores_array = pca.transform(X)
    score_labels = [f"PC{i + 1}" for i in range(scores_array.shape[1])]
    pc_labels = [f"PC{i + 1}" for i in range(pca.components_.shape[0])]
    loadings = pd.DataFrame(pca.components_, index=pc_labels, columns=feature_columns)
    scores = pd.DataFrame(scores_array, index=df.index, columns=score_labels)
    pve = pd.Series(pca.explained_variance_ratio_, index=pc_labels, name="prop_var")
    cpve = pd.Series(pve.cumsum(), index=pc_labels, name="cumulative_prop_var")
    return {
        "loadings": loadings,
        "scores": scores,
        "pve": pve,
        "cpve": cpve,
    }


def drop_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return df.drop(columns=[column])


def drop_row(df: pd.DataFrame, index_list: list[int]) -> pd.DataFrame:
    return df.drop(index=index_list).reset_index(drop=True)


def dotless_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    new_column = column.replace(".", "")
    return df.rename(columns={column: new_column})
