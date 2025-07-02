import typer
from pathlib import Path
from loguru import logger
from tennis_racquet_analysis.config import DATA_DIR, EXTERNAL_DATA_DIR
from tennis_racquet_analysis.preprocessing_utils import load_data
from tennis_racquet_analysis.cluster_prep_utils import (
    merge_cluster_labels,
    clusters_to_labels,
    count_labels,
    get_cluster_profiles,
)

app = typer.Typer()


@app.command("cluster-profiles")
def cluster_profiles(
    raw_input_file: str = typer.Argument(..., help="csv filename."),
    raw_input_dir: Path = typer.Option(
        "raw",
        "--input-dir",
        "-raw",
        help="Sub-folder under data/ (e.g. external, interim, processed, raw), where the input file lives.",
    ),
    cluster_input_file: str = typer.Argument(
        ...,
        help="csv filename.",
    ),
    cluster_input_dir: Path = typer.Option(
        "processed",
        "--input-dir",
        "-cluster",
        help="Sub-folder under data/ (e.g. external, interim, processed, raw), where the input file lives.",
    ),
    cluster_col: str = typer.Option(
        ...,
        "--cluster-col",
        "-c",
        help="Name of the column with your integer clusters (e.g. cluster_7).",
    ),
    key_column_name: str = typer.Option(
        None,
        "--key-col",
        "-k",
        help="(Optional) If both files share an ID column, merge on this instead of row order.",
    ),
):
    raw_input_path = DATA_DIR / raw_input_dir / raw_input_file
    raw_df = load_data(raw_input_path)
    cluster_input_path = DATA_DIR / cluster_input_dir / cluster_input_file
    cluster_df = load_data(cluster_input_path)
    if key_column_name:
        merged = raw_df.merge(
            cluster_df[[key_column_name, cluster_col]],
            on=key_column_name,
        )
    else:
        merged = merge_cluster_labels(
            raw_df,
            cluster_df=cluster_df,
            cluster_col=cluster_col,
        )
    cluster_profiles_df = get_cluster_profiles(
        df=merged,
        cluster_col=cluster_col,
    )
    raw_stem = Path(raw_input_file).stem
    output_file = f"{raw_stem}_{cluster_col}_profiles.csv"
    output_path = EXTERNAL_DATA_DIR / output_file
    cluster_profiles_df.to_csv(output_path, index=False)
    logger.success(f"Saved profiles to {output_path!r}")


@app.command("map-clusters")
def map_clusters(
    cluster_results_file: str = typer.Argument(
        ..., help="csv with cluster IDs from k-means model."
    ),
    input_dir: Path = typer.Option(
        "processed",
        "--input-dir",
        "-d",
        help="Sub-folder under data/ (e.g. external, interim, processed, raw), where the input file lives.",
    ),
    cluster_col: str = typer.Option(
        ..., "--cluster-col", "-c", help="Name of the raw cluster colum."
    ),
    output_dir: Path = typer.Option(
        EXTERNAL_DATA_DIR,
        "--output-dir:",
        "-o",
        dir_okay=True,
        file_okay=False,
    ),
):
    input_path = DATA_DIR / input_dir / cluster_results_file
    df_cluster = load_data(input_path)
    unique_ids = sorted(df_cluster[cluster_col].unique())
    mapping: dict[int, str] = {}
    for cluster_id in unique_ids:
        mapping[cluster_id] = typer.prompt(f"Label for cluster {cluster_id}")
    label_series = clusters_to_labels(
        df_cluster[cluster_col],
        mapping,
    )
    counts_df = count_labels(label_series, label_col="cluster_label")
    typer.echo("\nCluster ID -> Label Mapping:")
    for cluster_id, cluster_label in mapping.items():
        typer.echo(f"  {cluster_id} -> {cluster_label}")
    typer.echo("\nCounts per label:")
    typer.echo(counts_df.to_markdown(index=False))
    stem = Path(cluster_results_file).stem
    output_path = output_dir / f"{stem}_counts.csv"
    counts_df.to_csv(output_path, index=False)
    logger.success(f"Saved counts to {output_path!r}")


if __name__ == "__main__":
    app()
