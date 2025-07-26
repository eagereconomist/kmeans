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
      - 'loadings': DataFrame of shape (n_components, n_features)
      - 'scores':   DataFrame of shape (n_samples, n_components)
      - 'pve':      Series of length n_components with proportion of variance explained
      - 'cpve':     Series of length n_components with cumulative pve
    """
    # ─── pick features ───────────────────────────────────────────────
    if numeric_cols is None:
        all_num = df.select_dtypes(include="number").columns.tolist()
        feature_cols = [col for col in all_num if col != hue_column]
    else:
        # only keep those the user passed that are actually numeric
        feature_cols = [
            col
            for col in numeric_cols
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
        ]

    if not feature_cols:
        raise ValueError("No numeric features available for PCA.")

    # ─── extract matrix and fit PCA ────────────────────────────────
    X = df[feature_cols].values
    pca = PCA(n_components=n_components, random_state=random_state)
    scores_array = pca.fit_transform(X)

    # ─── wrap scores in a DataFrame with labeled PCs ───────────────
    scores = pd.DataFrame(
        scores_array,
        columns=[f"PC{i + 1}" for i in range(scores_array.shape[1])],
        index=df.index,
    )

    # ─── build loadings DataFrame (PC × features) ──────────────────
    loadings = pd.DataFrame(
        pca.components_,
        columns=feature_cols,
        index=[f"PC{i + 1}" for i in range(pca.components_.shape[0])],
    )

    # ─── proportion of variance explained ──────────────────────────
    prop_var = pd.Series(
        pca.explained_variance_ratio_,
        index=loadings.index,
        name="prop_var",
    )

    # ─── cumulative proportion ─────────────────────────────────────
    cum_var = prop_var.cumsum()
    cum_var.name = "cumulative_prop_var"

    return {
        "loadings": loadings,
        "scores": scores,
        "pve": prop_var,
        "cpve": cum_var,
    }
