import pandas as pd
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
    numeric_cols: list[str] | None = None,
    hue_column: str | None = None,
    n_components: int | None = None,
    random_state: int = 0,
) -> dict[str, pd.DataFrame | pd.Series]:
    """
    Returns a dict with:
      - 'loadings':    DataFrame of shape (n_components, n_features)
      - 'scores':      Array of shape (n_samples, n_components)
      - 'pve':         Series of length n_components with prop. var. explained
      - 'cpve':        Series of cumulative pve
    """
    # pick features
    if numeric_cols is None:
        all_num = df.select_dtypes(include="number").columns.tolist()
        feature_cols = [c for c in all_num if c != hue_column]
    else:
        feature_cols = numeric_cols

    if not feature_cols:
        raise ValueError("No numeric features available for PCA.")

    X = df[feature_cols].values

    pca = PCA(n_components=n_components, random_state=random_state)
    scores_array = pca.fit_transform(X)

    loadings = pd.DataFrame(
        pca.components_,
        columns=feature_cols,
        index=[f"PC{i + 1}" for i in range(pca.components_.shape[0])],
    )

    prop_var = pd.Series(pca.explained_variance_ratio_, index=loadings.index, name="prop_var")
    cum_var = prop_var.cumsum()
    cum_var.name = "cumulative_prop_var"

    return {
        "loadings": loadings,
        "scores": scores_array,
        "pve": prop_var,
        "cpve": cum_var,
    }


def drop_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.drop(columns=[col])


def drop_row(df: pd.DataFrame, index_list: list[int]) -> pd.DataFrame:
    return df.drop(index=index_list).reset_index(drop=True)


def dotless_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    new_col = col.replace(".", "")
    return df.rename(columns={col: new_col})
