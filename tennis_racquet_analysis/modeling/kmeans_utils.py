import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from typing import Optional, Sequence, Union, Tuple


def fit_kmeans(
    df: pd.DataFrame,
    k: int,
    feature_columns: Optional[Sequence[str]] = None,
    init: str = "k-means++",
    n_init: int = 50,
    random_state: int = 4572,
    algorithm: str = "lloyd",
    label_column: str = "cluster",
) -> pd.DataFrame:
    if feature_columns is None:
        X = df.select_dtypes(include=np.number).values
    else:
        X = df[list(feature_columns)].values
    km = KMeans(
        n_clusters=k, init=init, n_init=n_init, random_state=random_state, algorithm=algorithm
    )
    labels = km.fit_predict(X)
    df_km_labels = df.copy()
    df_km_labels[f"{label_column}_{k}"] = labels
    return df_km_labels


def batch_kmeans(
    df: pd.DataFrame,
    k_range: Union[Tuple[int, int], range] = (1, 10),
    feature_columns: Optional[Sequence[str]] = None,
    init: str = "k-means++",
    n_init: int = 50,
    random_state: int = 4572,
    algorithm: str = "lloyd",
    label_column: str = "cluster",
) -> pd.DataFrame:
    X = (
        df.select_dtypes(include=np.number).values
        if feature_columns is None
        else df[list(feature_columns)].values
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
        df_labeled[f"{label_column}_{k}"] = km.labels_
    return df_labeled
