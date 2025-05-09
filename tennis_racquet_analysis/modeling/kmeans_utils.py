from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
from typing import Optional, Sequence, Union, Tuple


def compute_kmeans_inertia(
    df: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
    k_range: Union[Tuple[int, int], range] = (1, 10),
    random_state: int = 4572,
) -> pd.DataFrame:
    if feature_columns is None:
        X = df.select_dtypes(include=np.number).values
    else:
        X = df[list(feature_columns)].values
    if isinstance(k_range, tuple):
        k_start, k_end = k_range
        ks = range(k_start, k_end + 1)
    else:
        ks = k_range
    inertia_vals = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state).fit(X)
        inertia_vals.append({"k": k, "inertia": km.inertia_})
        inertia_df = pd.DataFrame.from_records(inertia_vals)
    return inertia_df


def compute_silhouette_scores(
    df: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
    k_range: Union[Tuple[int, int], range] = (2, 10),
    random_state: int = 4572,
) -> pd.DataFrame:
    if feature_columns is None:
        X = df.select_dtypes(include=np.number).values
    else:
        X = df[list(feature_columns)].values
    n_samples = X.shape[0]
    if isinstance(k_range, tuple):
        k_start, k_end = k_range
        k_end = min(k_end, n_samples - 1)
        ks = range(k_start, k_end + 1)
    else:
        ks = k_range
    silhouette_vals = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state).fit(X)
        labels = km.labels_
        silhouette_scores = silhouette_score(X, labels)
        silhouette_vals.append({"n_clusters": k, "silhouette_score": silhouette_scores})
        silhouette_df = pd.DataFrame.from_records(silhouette_vals)
    return silhouette_df


def fit_kmeans(
    df: pd.DataFrame,
    k: int,
    feature_columns: Optional[Sequence[str]] = None,
    random_state: int = 4572,
    label_column: str = "cluster",
) -> pd.DataFrame:
    if feature_columns is None:
        X = df.select_dtypes(include=np.number).values
    else:
        X = df[list(feature_columns)].values
    km = KMeans(n_clusters=k, random_state=random_state)
    labels = km.fit_predict(X)
    df_km_labels = df.copy()
    df_km_labels[f"{label_column}_{k}"] = labels
    return df_km_labels


def batch_kmeans(
    df: pd.DataFrame,
    k_range: Union[Tuple[int, int], range] = (2, 10),
    feature_columns: Optional[Sequence[str]] = None,
    random_state: int = 4572,
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
        km = KMeans(n_clusters=k, random_state=random_state).fit(X)
        df_labeled[f"{label_column}_{k}"] = km.labels_
    return df_labeled
