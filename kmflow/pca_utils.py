import pandas as pd
from sklearn.decomposition import PCA


def compute_pca(
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
        feature_cols = [column for column in all_num if column != hue_column]
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
