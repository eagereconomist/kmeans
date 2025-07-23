import pandas as pd
from pandas.api import types as pd_types
from typing import Optional, Sequence, Dict


def merge_cluster_labels(
    raw_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    cluster_col: str,
) -> pd.DataFrame:
    """
    Append cluster_df[cluster_col] onto raw_df by row order.
    """
    if len(raw_df) != len(cluster_df):
        raise ValueError(f"Row counts differ ({len(raw_df)} vs. {len(cluster_df)})")
    merged = raw_df.copy().reset_index(drop=True)
    merged[cluster_col] = cluster_df.reset_index(drop=True)[cluster_col]
    return merged


def clusters_to_labels(
    cluster_ids: pd.Series,
    mapping: Dict[int, str],
) -> pd.Series:
    """
    Map integer cluster IDs to human labels.
    """
    return cluster_ids.map(mapping)


def count_labels(
    labels: pd.Series,
    label_col: str = "cluster_label",
) -> pd.DataFrame:
    """
    Count how many times each label appears.
    """
    counts = labels.value_counts().rename_axis(label_col).reset_index(name="count")
    return counts


def get_cluster_profiles(
    df: pd.DataFrame,
    cluster_col: str,
    numeric_cols: Optional[Sequence[str]] = None,
    stats: list[str] = ["mean", "median", "min", "max"],
) -> pd.DataFrame:
    """
    For each cluster, compute summary stats on the chosen features (or all numeric).
    """
    if numeric_cols is None:
        numeric_cols = [
            col for col in df.columns if col != cluster_col and pd_types.is_numeric_dtype(df[col])
        ]
    aggregated = df.groupby(cluster_col)[numeric_cols].agg(stats)
    aggregated.columns = [f"{feature}_{stat}" for feature, stat in aggregated.columns]
    return aggregated.reset_index()
