from pathlib import Path
from typing import Optional, Dict
from tqdm import tqdm
from loguru import logger
import typer

import kmflow.utils.cli_utils as cli_utils
import kmflow.utils.cluster_prep_utils as cluster_prep_utils

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
    # decide output path up front
    if output_file is None:
        stem = raw_file.stem if raw_file != Path("-") else "stdin"
        out_path = Path.cwd() / f"{stem}_{cluster_col}_profiles.csv"
    else:
        out_path = output_file

    with tqdm(total=4, desc="Cluster Profiles", colour="green") as pbar:
        # 1) read raw
        raw_df = cli_utils.read_df(raw_file)
        pbar.update(1)

        # 2) read cluster
        cluster_df = cli_utils.read_df(cluster_file)
        pbar.update(1)

        # 3) merge + compute
        if key_col:
            merged = raw_df.merge(
                cluster_df[[key_col, cluster_col]],
                on=key_col,
                how="inner",
            )
        else:
            merged = cluster_prep_utils.merge_cluster_labels(raw_df, cluster_df, cluster_col)
        profiles = cluster_prep_utils.get_cluster_profiles(merged, cluster_col)
        pbar.update(1)

        # 4) write out
        cli_utils._write_df(profiles, out_path)
        logger.success(
            f"Cluster profiles saved to {out_path!r}"
            if out_path != Path("-")
            else "Cluster profiles written to stdout."
        )
        pbar.update(1)


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
    df = cli_utils.read_df(cluster_file)
    unique_ids = sorted(df[cluster_col].unique())

    # 1) interactively build mapping
    mapping: Dict[int, str] = {}
    for cid in unique_ids:
        mapping[cid] = typer.prompt(f"Label for cluster {cid}")

    # 2) apply mapping & count
    labels = cluster_prep_utils.clusters_to_labels(df[cluster_col], mapping)
    counts = cluster_prep_utils.count_labels(labels, label_col="cluster_label")

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
    cli_utils._write_df(counts, output_file)


if __name__ == "__main__":
    app()
