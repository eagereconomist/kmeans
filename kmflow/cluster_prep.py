from pathlib import Path
from typing import Optional, Dict

import typer

from kmflow.cli_utils import read_df, _write_df
from kmflow.cluster_prep_utils import (
    merge_cluster_labels,
    clusters_to_labels,
    count_labels,
    get_cluster_profiles,
)

app = typer.Typer(help="Profile and label K-Means clusters.")


@app.command("cluster-profiles")
def cluster_profiles(
    raw_file: Path = typer.Argument(..., help="Raw CSV (use '-' for stdin)."),
    cluster_file: Path = typer.Argument(..., help="Clustered CSV (use '-' for stdin)."),
    cluster_col: str = typer.Argument(..., help="Column with cluster labels."),
    key_col: Optional[str] = typer.Option(
        None,
        "--key-col",
        "-k",
        help="If present, merge on this column instead of by row order.",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Where to write profiles CSV; use '-' for stdout (default).",
    ),
):
    """
    Generate per-cluster summary profiles.
    """
    # 1) read both tables
    raw_df = read_df(raw_file)
    cluster_df = read_df(cluster_file)

    # 2) merge
    if key_col:
        merged = raw_df.merge(
            cluster_df[[key_col, cluster_col]],
            on=key_col,
            how="inner",
        )
    else:
        merged = merge_cluster_labels(raw_df, cluster_df, cluster_col)

    # 3) compute profiles
    profiles = get_cluster_profiles(merged, cluster_col)

    # 4) write out
    if output_file is None:
        stem = raw_file.stem if raw_file != Path("-") else "stdin"
        output_file = Path.cwd() / f"{stem}_{cluster_col}_profiles.csv"
    _write_df(profiles, output_file)


@app.command("map-clusters")
def map_clusters(
    cluster_file: Path = typer.Argument(..., help="Clustered CSV (use '-' for stdin)."),
    cluster_col: str = typer.Argument(..., help="Column with cluster labels."),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Where to write counts CSV; use '-' for stdout (default).",
    ),
):
    """
    Prompt for human labels per cluster ID, then count.
    """
    df = read_df(cluster_file)
    unique_ids = sorted(df[cluster_col].unique())

    # 1) interactively build mapping
    mapping: Dict[int, str] = {}
    for cid in unique_ids:
        mapping[cid] = typer.prompt(f"Label for cluster {cid}")

    # 2) apply mapping & count
    labels = clusters_to_labels(df[cluster_col], mapping)
    counts = count_labels(labels, label_col="cluster_label")

    # 3) echo mapping & counts
    typer.echo("\nCluster -> Label mapping:")
    for cid, lab in mapping.items():
        typer.echo(f"  {cid} -> {lab}")

    typer.echo("\nCounts per label:")
    typer.echo(counts.to_markdown(index=False))

    # 4) write out
    if output_file is None:
        stem = cluster_file.stem if cluster_file != Path("-") else "stdin"
        output_file = Path.cwd() / f"{stem}_cluster_counts.csv"
    _write_df(counts, output_file)


if __name__ == "__main__":
    app()
