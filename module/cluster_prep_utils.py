import pandas as pd
from pandas.api import types as pd_types


def merge_cluster_labels(
    raw_df: pd.DataFrame, cluster_df: pd.DataFrame, cluster_col: str
) -> pd.DataFrame:
    if len(raw_df) != len(cluster_df):
        raise ValueError(f"Row counts differ ({len(raw_df)} vs. {len(cluster_df)})")
    merged = raw_df.copy().reset_index(drop=True)
    merged[cluster_col] = cluster_df.reset_index(drop=True)[cluster_col]
    return merged


def clusters_to_labels(
    cluster_ids: pd.Series,
    mapping: dict[int, str],
) -> pd.Series:
    return cluster_ids.map(mapping)


def count_labels(
    labels: pd.Series,
    label_col: str = "cluster_label",
) -> pd.DataFrame:
    counts = labels.value_counts().rename_axis(label_col).reset_index(name="count")
    return counts


def get_cluster_profiles(
    df: pd.DataFrame,
    cluster_col: str,
    feature_columns: list[str] | None = None,
    stats: list[str] = ["mean", "median", "min", "max"],
) -> pd.DataFrame:
    if feature_columns is None:
        feature_columns = [
            col for col in df.columns if col != cluster_col and pd_types.is_numeric_dtype(df[col])
        ]
    aggregated = df.groupby(cluster_col)[feature_columns].agg(stats)
    aggregated.columns = [f"{feature}_{stat}" for feature, stat in aggregated.columns]

    return aggregated.reset_index()
