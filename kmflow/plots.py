from typing import Optional, List
import sys
import typer
from tqdm import tqdm
from pathlib import Path
from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil


from kmflow.plots_utils import (
    _run_plot_with_progress,
    _apply_cubehelix_style,
    _ensure_unique_path,
    bar_plot,
    histogram,
    scatter_plot,
    box_plot,
    violin_plot,
    correlation_heatmap,
    qq_plot,
    inertia_plot,
    silhouette_plot,
    scree_plot,
    cumulative_var_plot,
    biplot,
    biplot_3d,
    cluster_scatter,
    cluster_scatter_3d,
    plot_batch_clusters,
)

from kmflow.preprocess_utils import compute_pca_summary

app = typer.Typer()


@app.command("barplot")
def barplot(
    input_file: Path = typer.Argument(..., help="Path to CSV file, or '-' to read from stdin."),
    category_col: str = typer.Argument(..., help="Categorical column (x-axis when vertical)."),
    numeric_col: str = typer.Argument(..., help="Numeric column to plot."),
    orientation: str = typer.Option("v", "--orientation", "-a", help="Orientation: 'v' or 'h'."),
    save: bool = typer.Option(
        True,
        "--no-save",
        "-n",
        help="Save to file (default) or display only.",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Path for the PNG output; use '-' to write image to stdout.",
    ),
):
    """
    Bar plot of <numeric_col> by <category_col>, with stdin/stdout or file I/O.
    """
    default_name = f"{category_col.capitalize()}_by_{numeric_col.capitalize()}_barplot.png"
    _run_plot_with_progress(
        name="Barplot",
        input_file=input_file,
        plot_fn=bar_plot,
        kwargs={
            "numeric_col": numeric_col,
            "category_col": category_col,
            "orient": orientation,
        },
        output_file=output_file,
        default_name=default_name,
        save=save,
    )


@app.command("histogram")
def hist(
    input_file: Path = typer.Argument(..., help="Path to CSV file, or '-' to read from stdin."),
    x_axis: str = typer.Argument(..., help="Column to histogram."),
    num_bins: int = typer.Option(10, "--bins", "-b", help="Number of bins."),
    save: bool = typer.Option(
        True,
        "--no-save",
        "-n",
        help="Save to file (default) or display only.",
    ),
    output_file: Path = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Path for the PNG output; use '-' to write image to stdout.",
    ),
):
    """
    Histogram of <x_axis> with optional stdin/stdout and save/display control.
    """
    default_name = f"{x_axis}_hist.png"

    _run_plot_with_progress(
        name="Histogram",
        input_file=input_file,
        plot_fn=histogram,
        kwargs={
            "num_bins": num_bins,
            "x_axis": x_axis,
        },
        output_file=output_file,
        default_name=default_name,
        save=save,
    )


@app.command("scatter")
def scatterplot(
    input_file: Path = typer.Argument(..., help="Path to CSV file, or '-' for stdin."),
    x_axis: str = typer.Argument(..., help="X-axis column."),
    y_axis: str = typer.Argument(..., help="Y-axis column."),
    scale: float = typer.Option(1.0, "--scale", "-s", help="Multiplier for x and y axes ranges."),
    save: bool = typer.Option(
        True, "--no-save", "-n", help="Save to file (default) or display only."
    ),
    output_file: Path = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Path for the PNG output; use '-' to write image to stdout.",
    ),
):
    """
    Scatter plot of x vs y, with optional stdin/stdout and save/display control.
    """
    default_name = f"{x_axis}_vs_{y_axis}_scatter.png"

    _run_plot_with_progress(
        name="Scatter plot",
        input_file=input_file,
        plot_fn=scatter_plot,
        kwargs={
            "x_axis": x_axis,
            "y_axis": y_axis,
            "scale": scale,
        },
        output_file=output_file,
        default_name=default_name,
        save=save,
    )


@app.command("boxplot")
def boxplot(
    input_file: Path = typer.Argument(..., help="Path to CSV file, or '-' to read from stdin."),
    category_col: Optional[str] = typer.Option(
        None, "--category-col", "-c", help="Column to group by (one box per category)."
    ),
    numeric_col: str = typer.Argument(..., help="Numeric column for the box plot."),
    patterns: Optional[List[str]] = typer.Option(
        None,
        "--pattern",
        "-p",
        help="Comma-separated regex pattern(s) to filter categories. For ex: 'Price, Quantity'",
    ),
    orientation: str = typer.Option("v", "--orientation", "-a", help="Orientation: 'v' or 'h'."),
    save: bool = typer.Option(
        True, "--no-save", "-n", help="Save to file (default) or display only."
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Path for the PNG output; use '-' to write image to stdout.",
    ),
):
    """
    Box plot of `numeric_col`, optionally grouped by `category_col` and filtered by `patterns`.
    """
    default_name = f"{(f'filtered_{category_col}' if patterns else (f'by_{category_col}' if category_col else 'all'))}_{numeric_col}_boxplot.png"  # new

    _run_plot_with_progress(
        name="Boxplot",
        input_file=input_file,
        plot_fn=box_plot,
        kwargs={
            "numeric_col": numeric_col,
            "category_col": category_col,
            "patterns": patterns[0].split(", ") if patterns else None,
            "orient": orientation,
        },
        output_file=output_file,
        default_name=default_name,
        save=save,
    )


@app.command("violin")
def violinplot(
    input_file: Path = typer.Argument(..., help="Path to CSV file, or '-' to read from stdin."),
    category_col: Optional[str] = typer.Option(
        None, "--category-col", "-c", help="Column to group by (one violin per category)."
    ),
    numeric_col: str = typer.Argument(..., help="Numeric column for the violin plot."),
    patterns: Optional[List[str]] = typer.Option(
        None,
        "--pattern",
        "-p",
        help="Comma-separated regex pattern(s) to filter categories. For ex: 'Price, Quantity'",
    ),
    orientation: str = typer.Option("v", "--orientation", "-a", help="Orientation: 'v' or 'h'."),
    inner: str = typer.Option(
        "box", "--inner", "-i", help="Interior representation inside the violins."
    ),
    save: bool = typer.Option(
        True, "--no-save", "-n", help="Save to file (default) or display only."
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Path for the PNG output; use '-' to write image to stdout.",
    ),
):
    """
    Violin plot of `numeric_col`, optionally grouped by `category_col` and filtered by `patterns`.
    """
    # build default filename based on flags
    default_name = (
        f"{'filtered_' + category_col if patterns else ('by_' + category_col if category_col else 'all')}"
        f"_{numeric_col}_violin.png"
    )

    # delegate to shared helper
    _run_plot_with_progress(
        name="Violin",
        input_file=input_file,
        plot_fn=violin_plot,
        kwargs={
            "numeric_col": numeric_col,
            "category_col": category_col,
            "patterns": patterns[0].split(", ") if patterns else None,
            "orient": orientation,
            "inner": inner,
        },
        output_file=output_file,
        default_name=default_name,
        save=save,
    )


@app.command("heatmap")
def corr_heatmap(
    input_file: Path = typer.Argument(..., help="Path to CSV file, or '-' to read from stdin."),
    save: bool = typer.Option(
        True, "--no-save", "-n", help="Save to file (default) or display only."
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Path for the PNG output; use '-' to write image to stdout.",
    ),
):
    """
    Correlation matrix heatmap for all numeric features in the data.
    """
    default_name = "heatmap.png"

    _run_plot_with_progress(
        name="Correlation Heatmap",
        input_file=input_file,
        plot_fn=correlation_heatmap,
        kwargs={},
        output_file=output_file,
        default_name=default_name,
        save=save,
    )


@app.command("qq")
def qq_plt(
    input_file: Path = typer.Argument(..., help="CSV file path or '-' to read from stdin."),
    numeric_col: Optional[str] = typer.Option(
        None,
        "--numeric-col",
        "-numeric-col",
        help="Numeric column for Q-Q plot, ex: 'Price, Quantity', (omit when using --all).",
    ),
    all_cols: bool = typer.Option(
        False, "--all", "-a", help="Generate Q-Q plots for all numeric columns."
    ),
    save: bool = typer.Option(
        True, "--no-save", "-n", help="Save to file (default) or display only."
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output-file", "-o", help="Path for PNG output; use '-' for stdout."
    ),
):
    """
    Generate a Q-Q plot for one numeric column or all numeric columns.
    """
    # ─── Multi-column mode ────────────────────────────────────────────────
    if all_cols:
        df = pd.read_csv(sys.stdin) if input_file == Path("-") else pd.read_csv(input_file)
        cols = df.select_dtypes(include="number").columns.tolist()
        if not cols:
            raise typer.BadParameter("No numeric columns found.")

        with tqdm(total=2, desc="Q-Q Plots", colour="green") as pbar:
            # 1) draw grid
            _apply_cubehelix_style()
            n = len(cols)
            ncols = 3
            nrows = ceil(n / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
            axes_flat = axes.flatten()
            for i, col in enumerate(cols):
                # draw but don't save yet
                qq_plot(df, col, output_path=None, save=False, ax=axes_flat[i])
                axes_flat[i].set_title(col)
            for ax in axes_flat[n:]:
                ax.set_visible(False)
            fig.suptitle("Q-Q Plots")
            fig.tight_layout()
            pbar.update(1)

            # 2) save/stream/show
            default_name = "qq_all.png"
            if output_file == Path("-"):
                fig.savefig(sys.stdout.buffer, format="png")
                logger.success("Q-Q plots PNG written to stdout.")
            elif not save:
                plt.show()
                plt.close(fig)
                logger.success("Q-Q plots displayed (not saved).")
            else:
                out = _ensure_unique_path(output_file or Path.cwd() / default_name)
                fig.savefig(out)
                plt.close(fig)
                logger.success(f"Q-Q plots saved to {out!r}")
            pbar.update(1)

        return

    # ─── Single-column mode ───────────────────────────────────────────────
    if not numeric_col:
        raise typer.BadParameter("Specify a column via argument or use --all.")

    default_name = f"{numeric_col}_qq.png"
    _run_plot_with_progress(
        name=f"Q-Q Plot: {numeric_col}",
        input_file=input_file,
        plot_fn=qq_plot,
        kwargs={"numeric_col": numeric_col},
        output_file=output_file,
        default_name=default_name,
        save=save,
    )


@app.command("inertia")
def inertia(
    input_file: Path = typer.Argument(
        ..., help="CSV file of inertia vs k, or '-' to read from stdin."
    ),
    save: bool = typer.Option(
        True,
        "--no-save",
        "-n",
        help="Save to file (default) or display only.",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Output PNG path; use '-' to write image to stdout.",
    ),
):
    """
    Elbow plot of K-Means inertia versus number of clusters.
    """
    default_name = "inertia.png"

    _run_plot_with_progress(
        name="Inertia",
        input_file=input_file,
        plot_fn=lambda df, output_path, save: inertia_plot(
            inertia_df=df,
            output_path=output_path,
            save=save,
        ),
        kwargs={},
        output_file=output_file,
        default_name=default_name,
        save=save,
    )


@app.command("silhouette")
def silhouette(
    input_file: Path = typer.Argument(
        ..., help="CSV file of silhouette scores or '-' to read from stdin."
    ),
    save: bool = typer.Option(
        True, "--no-save", "-n", help="Save to file (default) or display only."
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Output PNG path; use '-' to write image to stdout.",
    ),
):
    """
    Plot silhouette score versus number of clusters.
    """
    default_name = "silhouette.png"

    _run_plot_with_progress(
        name="Silhouette",
        input_file=input_file,
        plot_fn=lambda df, output_path, save: silhouette_plot(
            silhouette_df=df, output_path=output_path, save=save
        ),
        kwargs={},
        output_file=output_file,
        default_name=default_name,
        save=save,
    )


@app.command("scree")
def scree(
    input_file: Path = typer.Argument(
        ..., help="CSV file of PCA variance or '-' to read from stdin."
    ),
    save: bool = typer.Option(
        True, "--no-save", "-n", help="Save to file (default) or display only."
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Path for the PNG output; use '-' to write image to stdout.",
    ),
):
    """
    Scree plot of proportion variance explained by each principal component.
    """
    default_name = "scree.png"

    _run_plot_with_progress(
        name="Proportion of Variance (Scree)",
        input_file=input_file,
        plot_fn=lambda df, output_path, save: scree_plot(
            df=df,
            output_path=output_path,
            save=save,
        ),
        kwargs={},
        output_file=output_file,
        default_name=default_name,
        save=save,
    )


@app.command("cpv")
def cpv(
    input_file: Path = typer.Argument(
        ..., help="CSV file of PCA cumulative variance or '-' to read from stdin."
    ),
    save: bool = typer.Option(
        True, "--no-save", "-n", help="Save to file (default) or display only."
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Path for the PNG output; use '-' to write image to stdout.",
    ),
):
    """
    Plot cumulative proportion of variance explained by principal components.
    """
    default_name = "cumulative_prop_var.png"

    _run_plot_with_progress(
        name="Proportion of Cumulative Variance",
        input_file=input_file,
        plot_fn=cumulative_var_plot,
        kwargs={},
        output_file=output_file,
        default_name=default_name,
        save=save,
    )


@app.command("cluster")
def cluster(
    input_file: Path = typer.Argument(
        ...,
        help="CSV of clustered data or '-' to read from stdin.",
    ),
    x_axis: Optional[str] = typer.Argument(..., help="Feature for X axis."),
    y_axis: Optional[str] = typer.Argument(..., help="Feature for Y axis."),
    cluster_col: str = typer.Argument(..., help="Column with cluster labels."),
    scale: float = typer.Option(1.0, "--scale", "-s", help="Multiplier for x/y axes ranges."),
    save: bool = typer.Option(
        True, "--no-save", "-n", help="Save to file (default) or display only."
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Output PNG path; use '-' for stdout.",
    ),
):
    """
    Scatter plot of X vs. Y colored by cluster labels.
    """
    if input_file == Path("-"):
        df = pd.read_csv(sys.stdin)
    else:
        df = pd.read_csv(input_file)
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    x_col = x_axis or numeric_columns[0]
    y_col = y_axis or (numeric_columns[1] if len(numeric_columns) > 1 else numeric_columns[0])
    if output_file is None:
        output_file = Path.cwd() / f"{x_col}_vs_{y_col}_cluster.png"
    output_file = _ensure_unique_path(output_file)
    ax = cluster_scatter(
        df=df,
        x_axis=x_col,
        y_axis=y_col,
        cluster_col=cluster_col,
        output_path=output_file,
        scale=scale,
        save=save,
    )
    fig = ax.figure
    if output_file == Path("-"):
        fig.savefig(sys.stdout.buffer, format="png")
        logger.success("Cluster scatter PNG written to stdout.")
    elif save:
        fig.savefig(output_file)
        plt.close(fig)
        logger.success(f"Cluster scatter saved to {output_file!r}")
    else:
        plt.show()
        plt.close(fig)
        logger.success("Cluster scatter displayed (not saved).")


@app.command("3d-cluster")
def cluster3d(
    input_file: Path = typer.Argument(..., help="Clustered CSV file, or '-' to read from stdin."),
    numeric_cols: Optional[List[str]] = typer.Argument(
        ...,
        help="Exactly three numeric columns (e.g. 'weight' 'height' 'width'); defaults to the first three numeric columns if omitted.",
    ),
    cluster_col: str = typer.Argument(..., help="Column with cluster labels."),
    scale: float = typer.Option(
        1.0, "--scale", "-s", help="Multiplier for x, y, and z axis ranges."
    ),
    save: bool = typer.Option(
        True,
        "--no-save",
        "-n",
        help="Save to file (default) or display only.",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Where to save the PNG; use '-' to write image to stdout.",
    ),
):
    """
    3D scatter of three features colored by cluster labels.
    """
    if input_file == Path("-"):
        df = pd.read_csv(sys.stdin)
    else:
        df = pd.read_csv(input_file)
    cols = numeric_cols or df.select_dtypes(include="number").columns[:3].tolist()
    if len(cols) != 3:
        raise typer.BadParameter("Must specify exactly three numeric columns for 3D.")
    if output_file is None:
        output_file = Path.cwd() / "cluster_3d.png"
    output_file = _ensure_unique_path(output_file)
    fig = cluster_scatter_3d(
        df=df,
        numeric_cols=cols,
        cluster_col=cluster_col,
        scale=scale,
        output_path=output_file,
        save=save,
    )
    fig.update_traces(marker=dict(size=5, opacity=1))
    fig.update_layout(
        legend_title_text="Cluster",
        scene=dict(
            xaxis_title=cols[0],
            yaxis_title=cols[1],
            zaxis_title=cols[2],
        ),
    )
    if output_file == Path("-"):
        fig.write_image(sys.stdout.buffer, format="png")
        logger.success("3D cluster scatter PNG written to stdout.")
    elif save:
        fig.write_image(str(output_file))
        logger.success(f"3D cluster scatter saved to {output_file!r}")
    else:
        fig.show(renderer="browser")
        logger.success("3D cluster scatter opened in browser (not saved).")


@app.command("cluster-subplot")
def batch_cluster_plot(
    input_file: Path = typer.Argument(..., help="Clustered CSV file, or '-' to read from stdin."),
    x_axis: Optional[str] = typer.Argument(
        ...,
        help="Feature for X axis.",
    ),
    y_axis: Optional[str] = typer.Argument(
        ...,
        help="Feature for Y axis.",
    ),
    cluster_col: str = typer.Option(
        "cluster_",
        "--cluster-col",
        "-cluster-col",
        help="Name of the column containing clusters in DataFrame from input_file",
    ),
    scale: float = typer.Option(1.0, "--scale", "-s", help="Multiplier for x and y axis ranges."),
    save: bool = typer.Option(
        True,
        "--no-save",
        "-n",
        help="Save to file (default) or display only.",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Where to save the PNG; use '-' to write image to stdout.",
    ),
):
    if input_file == Path("-"):
        df = pd.read_csv(sys.stdin)
    else:
        df = pd.read_csv(input_file)
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    if not numeric_columns:
        raise typer.BadParameter("No numeric columns found in your data.")
    if output_file is None:
        output_file = Path.cwd() / "cluster_subplot.png"
    output_file = _ensure_unique_path(output_file)
    x_col = x_axis or numeric_columns[0]
    y_col = y_axis or (numeric_columns[1] if len(numeric_columns) > 1 else numeric_columns[0])
    sorted_cluster_col = sorted(
        (column for column in df.columns if column.startswith(cluster_col)),
        key=lambda c: int(c.replace(cluster_col, "")),
    )
    if not sorted_cluster_col:
        raise typer.BadParameter(f"No columns found with prefix {cluster_col!r}")
    if output_file is None:
        output_file = Path.cwd() / f"{x_col}_vs_{y_col}_batch.png"
    output_file = _ensure_unique_path(output_file)
    fig = plot_batch_clusters(
        df,
        x_axis=x_col,
        y_axis=y_col,
        cluster_cols=sorted_cluster_col,
        output_path=output_file,
        save=False,
        scale=scale,
    )
    if output_file == Path("-"):
        fig.savefig(sys.stdout.buffer, format="png")
        logger.success("Cluster subplot PNG written to stdout.")
    elif save:
        fig.savefig(str(output_file))
        logger.success(f"Cluster subplot saved to {output_file!r}")
    else:
        plt.show()
        plt.close(fig)
        logger.success("Cluster subplot displayed (not saved).")


@app.command("biplot")
def plot_pca_biplot(
    input_file: Path = typer.Argument(..., help="CSV file to read (use '-' to read from stdin)."),
    numeric_cols: List[str] = typer.Option(
        None,
        "--numeric-cols",
        "-numeric-cols",
        help="Defaults to all numeric columns if omitted. Option to pick the numeric columns (e.g. 'weight' 'height' 'width', etc.).",
    ),
    compute_scores: bool = typer.Option(
        True,
        "--skip-scores",
        "-skip-scores",
        help="By default, compute PC scores from raw features; if skipped, assume df already contains PC columns.",
    ),
    pc_x: int = typer.Option(
        0, "--pc-x", "-x", help="Principal component for x-axis (0-indexed)."
    ),
    pc_y: int = typer.Option(
        1, "--pc-y", "-y", help="Principal component for y-axis (0-indexed)."
    ),
    scale: float = typer.Option(
        1.0, "--scale", "-s", help="Arrow length multiplier for loadings."
    ),
    hue_column: Optional[str] = typer.Option(
        None,
        "--hue-column",
        "-hue",
        help="Column name for coloring samples (will be excluded loading vectors).",
    ),
    save: bool = typer.Option(
        True,
        "--no-save",
        "-n",
        help="Save static PNG (default) or just display it.",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Where to save the PNG; use '-' to write image to stdout.",
    ),
):
    if input_file == Path("-"):
        df = pd.read_csv(sys.stdin)
    else:
        df = pd.read_csv(input_file)
    summary = compute_pca_summary(df=df, feature_columns=numeric_cols, hue_column=hue_column)
    loadings = summary["loadings"]
    pve = summary["pve"]
    hue = df[hue_column] if hue_column else None
    fig = biplot(
        df=df,
        loadings=loadings,
        pve=pve,
        compute_scores=compute_scores,
        pc_x=pc_x,
        pc_y=pc_y,
        scale=scale,
        hue=hue,
        save=save,
        output_path=output_file,
    )
    if output_file is None:
        output_file = Path("biplot.png")
    output_file = _ensure_unique_path(output_file)
    if output_file == Path("-"):
        fig.savefig(sys.stdout.buffer, format="png")
        logger.success("Biplot PNG written to stdout.")
    elif save:
        fig.savefig(str(output_file))
        logger.success(f"Biplot saved to {output_file!r}")
    else:
        plt.show()
        plt.close()
        logger.success("Biplot displayed (not saved).")


@app.command("3d-biplot")
def plot_3d_biplot(
    input_file: Path = typer.Argument(..., help="CSV file to read (use '-' to read from stdin)."),
    numeric_cols: Optional[List[str]] = typer.Argument(
        None,
        help="Exactly three numeric columns (e.g. 'weight' 'height' 'width'); defaults to all numeric columns if omitted.",
    ),
    hue_column: Optional[str] = typer.Option(
        None,
        "--hue-column",
        "-h",
        help="Column name for coloring samples (excluded from loadings).",
    ),
    compute_scores: bool = typer.Option(
        True,
        "--skip-scores",
        "-skip-scores",
        help="If set, skip recomputing PC scores (assume df already has PC columns).",
    ),
    pc_x: int = typer.Option(
        0,
        "--pc-x",
        "-x",
        help="Index of principal component for X axis (0-based).",
    ),
    pc_y: int = typer.Option(
        1,
        "--pc-y",
        "-y",
        help="Index of principal component for Y axis (0-based).",
    ),
    pc_z: int = typer.Option(
        2,
        "--pc-z",
        "-z",
        help="Index of principal component for Z axis (0-based).",
    ),
    scale: float = typer.Option(
        1.0,
        "--scale",
        "-s",
        help="Multiplier for loading arrow lengths.",
    ),
    save: bool = typer.Option(
        True,
        "--no-save",
        "-n",
        help="Save static PNG (default) or display interactive plot.",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Where to save the PNG; use '-' to write image bytes to stdout.",
    ),
):
    """
    3D PCA biplot: scores + loading vectors in three dimensions.
    """
    if input_file == Path("-"):
        df = pd.read_csv(sys.stdin)
    else:
        df = pd.read_csv(input_file)
    summary = compute_pca_summary(
        df=df,
        feature_columns=numeric_cols,
        hue_column=hue_column,
    )
    loadings, pve = summary["loadings"], summary["pve"]
    hue = df[hue_column] if hue_column else None
    if output_file is None:
        output_file = Path.cwd() / "3d_biplot.png"
    output_file = _ensure_unique_path(output_file)
    fig = biplot_3d(
        df=df,
        loadings=loadings,
        pve=pve,
        compute_scores=compute_scores,
        pc_x=pc_x,
        pc_y=pc_y,
        pc_z=pc_z,
        scale=scale,
        hue=hue,
        output_path=output_file,
        save=save,
    )
    if output_file == Path("-"):
        fig.write_image(sys.stdout.buffer, format="png")
        logger.success("3D biplot PNG written to stdout.")
    elif save:
        fig.write_image(str(output_file))
        logger.success(f"3D biplot saved to {output_file!r}")
    else:
        fig.show(renderer="browser")
        logger.success("3D biplot opened in browser (not saved).")


if __name__ == "__main__":
    app()
