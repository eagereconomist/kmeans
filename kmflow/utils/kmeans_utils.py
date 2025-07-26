import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from typing import Optional, Sequence, Union, Tuple


def fit_kmeans(
    df: pd.DataFrame,
    k: int,
    numeric_cols: Optional[Sequence[str]] = None,
    init: str = "k-means++",
    n_init: int = 50,
    random_state: int = 4572,
    algorithm: str = "lloyd",
    cluster_col: str = "cluster",
) -> pd.DataFrame:
    """
    Fit K-Means on the numeric columns the user passed (or all numeric if none),
    but still carry through any non-numeric columns passed into the output.
    """
    # ─── split out what the user asked for ──────────────────────────────────
    if numeric_cols:
        requested = list(numeric_cols)
        numeric_feats = [col for col in requested if pd.api.types.is_numeric_dtype(df[col])]
        passthrough_feats = [col for col in requested if col not in numeric_feats]
    else:
        numeric_feats = df.select_dtypes(include=np.number).columns.tolist()
        passthrough_feats = []

    # ─── build the matrix only from the numeric features ────────────────
    X = df[numeric_feats].values

    km = KMeans(
        n_clusters=k,
        init=init,
        n_init=n_init,
        random_state=random_state,
        algorithm=algorithm,
    )
    labels = km.fit_predict(X)

    # ─── assemble output: requested string cols + numeric inputs + cluster ──
    output_cols = passthrough_feats + numeric_feats
    df_km_labels = df[output_cols].copy()
    df_km_labels[f"{cluster_col}_{k}"] = labels

    return df_km_labels


def batch_kmeans(
    df: pd.DataFrame,
    k_range: Union[Tuple[int, int], range] = (1, 20),
    numeric_cols: Optional[Sequence[str]] = None,
    init: str = "k-means++",
    n_init: int = 50,
    random_state: int = 4572,
    algorithm: str = "lloyd",
    cluster_col: str = "cluster",
) -> pd.DataFrame:
    """
    Run K-Means for each k in k_range on the numeric columns the user passed (or all numeric if none),
    but carry through any non-numeric columns too.
    """
    # ─── split requested columns into numeric vs passthrough ─────────────
    if numeric_cols:
        requested = list(numeric_cols)
        numeric_feats = [col for col in requested if pd.api.types.is_numeric_dtype(df[col])]
        passthrough_feats = [col for col in requested if col not in numeric_feats]
    else:
        numeric_feats = df.select_dtypes(include=np.number).columns.tolist()
        passthrough_feats = []

    # ─── build X from numeric features only ───────────────────────────────
    X = df[numeric_feats].values

    # ─── expand k_range into an iterable of ints ─────────────────────────
    if isinstance(k_range, tuple):
        k_start, k_end = k_range
        ks = range(k_start, k_end + 1)
    else:
        ks = k_range

    # ─── prepare output DataFrame with only requested cols ────────────────
    output_cols = passthrough_feats + numeric_feats
    df_labeled = df[output_cols].copy()

    # ─── loop over each k, fit & label ─────────────────────────────────
    for k in ks:
        algo_option = algorithm if k > 1 else "lloyd"
        km = KMeans(
            n_clusters=k,
            init=init,
            n_init=n_init,
            random_state=random_state,
            algorithm=algo_option,
        ).fit(X)
        df_labeled[f"{cluster_col}_{k}"] = km.labels_

    return df_labeled
