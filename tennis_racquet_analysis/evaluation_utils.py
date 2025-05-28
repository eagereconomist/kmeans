import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from typing import Optional, Sequence, Union, Tuple, Iterable


def compute_inertia_scores(
    df: pd.DataFrame,
    k_range: Union[Tuple[int, int], range] = (1, 20),
    feature_columns: Optional[Sequence[str]] = None,
    init: str = "k-means++",
    n_init: int = 50,
    random_state: int = 4572,
    algorithm: str = "lloyd",
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
    inertia_vals = []
    for k in ks:
        algo_option = algorithm if k > 1 else "lloyd"
        km = KMeans(
            n_clusters=k,
            init=init,
            n_init=n_init,
            random_state=random_state,
            algorithm=algo_option,
        ).fit(X)
        inertia_vals.append({"k": k, "inertia": km.inertia_})
    return pd.DataFrame.from_records(inertia_vals)


def compute_silhouette_scores(
    df: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
    init: str = "k-means++",
    n_init: int = 50,
    random_state: int = 4572,
    algorithm: str = "lloyd",
    k_values: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    X = (
        df.select_dtypes(include=np.number).values
        if feature_columns is None
        else df[list(feature_columns)].values
    )
    n_samples = X.shape[0]
    ks = k_values if k_values is not None else range(2, n_samples)
    silhouette_vals = []
    for k in ks:
        km = KMeans(
            n_clusters=k, init=init, n_init=n_init, random_state=random_state, algorithm=algorithm
        ).fit(X)
        labels = km.labels_
        silhouette_scores = silhouette_score(X, labels)
        silhouette_vals.append({"n_clusters": k, "silhouette_score": silhouette_scores})
    return pd.DataFrame.from_records(silhouette_vals)


def compute_calinski_scores(
    df: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
    init: str = "k-means++",
    n_init: int = 50,
    random_state: int = 4572,
    algorithm: str = "lloyd",
    k_values: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    X = (
        df.select_dtypes(include=np.number).values
        if feature_columns is None
        else df[list(feature_columns)].values
    )
    n_samples = X.shape[0]
    ks = k_values if k_values is not None else range(2, n_samples)
    calinski_vals = []
    for k in ks:
        km = KMeans(
            n_clusters=k,
            init=init,
            n_init=n_init,
            random_state=random_state,
            algorithm=algorithm,
        ).fit(X)
        labels = km.labels_
        calinski_scores = calinski_harabasz_score(X, labels)
        calinski_vals.append({"n_clusters": int(k), "calinski": calinski_scores})
    return pd.DataFrame.from_records(calinski_vals)


def compute_davies_scores(
    df: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
    init: str = "k-means++",
    n_init: int = 50,
    random_state: int = 4572,
    algorithm: str = "lloyd",
    k_values: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    X = (
        df.select_dtypes(include=np.number).values
        if feature_columns is None
        else df[list(feature_columns)].values
    )
    n_samples = X.shape[0]
    ks = k_values if k_values is not None else range(2, n_samples)
    records = []
    for k in ks:
        km = KMeans(
            n_clusters=k,
            init=init,
            n_init=n_init,
            random_state=random_state,
            algorithm=algorithm,
        ).fit(X)
        labels = km.labels_
        score = davies_bouldin_score(X, labels)
        records.append({"n_clusters": int(k), "davies": score})
    return pd.DataFrame.from_records(records)
