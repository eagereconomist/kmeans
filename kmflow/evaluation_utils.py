import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from typing import Optional, Sequence, Union, Tuple, Iterable
from pathlib import Path
from functools import reduce
import re


def load_metric_results(processed_root: Path, metric: str) -> pd.DataFrame:
    records = []

    for variant_dir in processed_root.iterdir():
        if not variant_dir.is_dir():
            continue
        variant = variant_dir.name

        for kmeans_root in variant_dir.iterdir():
            if not kmeans_root.is_dir() or not kmeans_root.name.startswith("kmeans"):
                continue

            for algo_dir in kmeans_root.iterdir():
                if not algo_dir.is_dir() or not algo_dir.name.startswith("algo_"):
                    continue

                expression = re.match(r"^algo_([^_]+)_init_(.+)$", algo_dir.name)
                if not expression:
                    continue
                algorithm, init = expression.group(1), expression.group(2)

                for path in algo_dir.glob(f"*_{metric}.csv"):
                    stem = path.stem
                    input_stem = stem[: -len(f"_{metric}")]

                    df = pd.read_csv(path)
                    df = df.rename(columns={df.columns[0]: "n_clusters", df.columns[1]: metric})
                    df["variant"] = variant
                    df["algorithm"] = algorithm
                    df["init"] = init
                    df["input_stem"] = input_stem
                    df = df[["variant", "algorithm", "init", "input_stem", "n_clusters", metric]]
                    records.append(df)
    if records:
        return pd.concat(records, ignore_index=True)
    else:
        return pd.DataFrame(
            columns=["variant", "algorithm", "init", "input_stem", "n_clusters", metric]
        )


def load_calinski_results(processed_root: Path) -> pd.DataFrame:
    return load_metric_results(processed_root, "calinski")


def load_davies_results(processed_root: Path) -> pd.DataFrame:
    return load_metric_results(processed_root, "davies")


def merge_benchmarks(
    calinski_df: pd.DataFrame,
    davies_df: pd.DataFrame,
) -> pd.DataFrame:
    calinski = calinski_df.rename(columns={"input_stem": "stem_calinski"})
    davies = davies_df.rename(columns={"input_stem": "stem_davies"})

    calinski = calinski[
        ["variant", "algorithm", "init", "n_clusters", "calinski", "stem_calinski"]
    ]
    davies = davies[["variant", "algorithm", "init", "n_clusters", "davies", "stem_davies"]]

    merged = reduce(
        lambda left, right: pd.merge(
            left, right, on=["variant", "algorithm", "init", "n_clusters"], how="outer"
        ),
        [calinski, davies],
    )

    merged["input_stem"] = merged["stem_calinski"].fillna(merged["stem_davies"])

    return merged[
        [
            "variant",
            "algorithm",
            "init",
            "input_stem",
            "n_clusters",
            "calinski",
            "davies",
        ]
    ]


def compute_inertia_scores(
    df: pd.DataFrame,
    k_range: Union[Tuple[int, int], range] = (1, 20),
    numeric_cols: Optional[Sequence[str]] = None,
    init: str = "k-means++",
    n_init: int = 50,
    random_state: int = 4572,
    algorithm: str = "lloyd",
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
    numeric_cols: Optional[Sequence[str]] = None,
    init: str = "k-means++",
    n_init: int = 50,
    random_state: int = 4572,
    algorithm: str = "lloyd",
    k_values: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    X = (
        df.select_dtypes(include=np.number).values
        if numeric_cols is None
        else df[list(numeric_cols)].values
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
    numeric_cols: Optional[Sequence[str]] = None,
    init: str = "k-means++",
    n_init: int = 50,
    random_state: int = 4572,
    algorithm: str = "lloyd",
    k_values: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    X = (
        df.select_dtypes(include=np.number).values
        if numeric_cols is None
        else df[list(numeric_cols)].values
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
    numeric_cols: Optional[Sequence[str]] = None,
    init: str = "k-means++",
    n_init: int = 50,
    random_state: int = 4572,
    algorithm: str = "lloyd",
    k_values: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    X = (
        df.select_dtypes(include=np.number).values
        if numeric_cols is None
        else df[list(numeric_cols)].values
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
