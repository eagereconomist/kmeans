import pandas as pd
from typing import Optional, Sequence, Dict, Union
from sklearn.decomposition import PCA


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
    numeric_col: Optional[Sequence[str]] = None,
    hue_column: Optional[str] = None,
    n_components: Optional[int] = None,
    random_state: int = 4572,
) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
    if numeric_col is None:
        all_numeric_cols = df.select_dtypes(include="number").columns.tolist()
        numeric_col = [c for c in all_numeric_cols if c != hue_column]
    X = df[numeric_col].values
    pca = PCA(n_components=n_components, random_state=random_state).fit(X)
    scores_array = pca.transform(X)
    score_labels = [f"PC{i + 1}" for i in range(scores_array.shape[1])]
    pc_labels = [f"PC{i + 1}" for i in range(pca.components_.shape[0])]
    loadings = pd.DataFrame(pca.components_, index=pc_labels, columns=numeric_col)
    scores = pd.DataFrame(scores_array, index=df.index, columns=score_labels)
    pve = pd.Series(pca.explained_variance_ratio_, index=pc_labels, name="prop_var")
    cpve = pd.Series(pve.cumsum(), index=pc_labels, name="cumulative_prop_var")
    return {
        "loadings": loadings,
        "scores": scores,
        "pve": pve,
        "cpve": cpve,
    }


def drop_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.drop(columns=[col])


def drop_row(df: pd.DataFrame, index_list: list[int]) -> pd.DataFrame:
    return df.drop(index=index_list).reset_index(drop=True)


def dotless_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    new_col = col.replace(".", "")
    return df.rename(columns={col: new_col})
