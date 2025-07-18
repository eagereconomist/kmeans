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
    if numeric_cols is None:
        X = df.select_dtypes(include=np.number).values
    else:
        X = df[list(numeric_cols)].values
    km = KMeans(
        n_clusters=k, init=init, n_init=n_init, random_state=random_state, algorithm=algorithm
    )
    labels = km.fit_predict(X)
    df_km_labels = df.copy()
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
    X = (
        df.select_dtypes(include=np.number).values
        if numeric_cols is None
        else df[list(numeric_cols)].values
    )
    if isinstance(k_range, tuple):
        k_start, k_end = k_range
        ks = range(k_start, k_end + 1)
    else:
        ks = k_range
    df_labeled = df.copy()
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
