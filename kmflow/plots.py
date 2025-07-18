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
        ..., help="CSV of clustered data or '-' to read from stdin."
    ),
    x_axis: str = typer.Argument(..., help="Feature for X axis."),
    y_axis: str = typer.Argument(..., help="Feature for Y axis."),
    cluster_col: str = typer.Argument(..., help="Column with cluster labels."),
    scale: float = typer.Option(1.0, "--scale", "-s", help="Multiplier for x/y axes ranges."),
    save: bool = typer.Option(
        True, "--no-save", "-n", help="Save to file (default) or display only."
    ),
    output_file: Path = typer.Option(
        None, "--output-file", "-o", help="Path for the PNG output; use '-' for stdout."
    ),
):
    """
    Scatter plot of X vs. Y colored by cluster labels.
    """
    # ─── 1) Load ─────────────────────────────────────────────────────────────────────────
    df = pd.read_csv(sys.stdin) if input_file == Path("-") else pd.read_csv(input_file)
    numeric = df.select_dtypes("number").columns.tolist()
    x_col = x_axis or numeric[0]
    y_col = y_axis or (numeric[1] if len(numeric) > 1 else numeric[0])

    default_name = f"{x_col}_vs_{y_col}_cluster.png"

    # ─── 2) Interactive only ───────────────────────────────────────────────
    if not save:
        ax = cluster_scatter(
            df=df,
            x_axis=x_col,
            y_axis=y_col,
            cluster_col=cluster_col,
            scale=scale,
            save=False,
            output_path=Path("unused.png"),
        )
        plt.show()
        plt.close(ax.figure)
        logger.success("Cluster scatter displayed (not saved).")
        return

    # ─── 3) Determine output path ───────────────────────────────────────────
    if output_file == Path("-"):
        # stream PNG to stdout
        ax = cluster_scatter(
            df=df,
            x_axis=x_col,
            y_axis=y_col,
            cluster_col=cluster_col,
            scale=scale,
            save=False,
            output_path=Path("unused.png"),
        )
        fig = ax.figure
        fig.savefig(sys.stdout.buffer, format="png")
        logger.success("Cluster scatter PNG written to stdout.")
    else:
        # write to disk
        out = output_file or (Path.cwd() / default_name)
        out = _ensure_unique_path(out)
        cluster_scatter(
            df=df,
            x_axis=x_col,
            y_axis=y_col,
            cluster_col=cluster_col,
            scale=scale,
            save=True,
            output_path=out,
        )
        logger.success(f"Cluster scatter saved to {out!r}")


@app.command("3d-cluster")
def cluster3d(
    input_file: Path = typer.Argument(..., help="Clustered CSV file, or '-' to read from stdin."),
    numeric_cols: Optional[List[str]] = typer.Argument(
        None,
        help="Exactly three numeric columns (e.g. 'weight' 'height' 'width'); "
        "if omitted, defaults to the first three numeric columns.",
    ),
    cluster_col: str = typer.Argument(..., help="Column with cluster labels."),
    scale: float = typer.Option(
        1.0, "--scale", "-s", help="Multiplier for x, y, and z axis ranges."
    ),
    save: bool = typer.Option(
        True, "--no-save", "-n", help="Save to file (default) or display only."
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Where to save the PNG; use '-' to write image bytes to stdout.",
    ),
):
    """
    3D scatter of three features colored by cluster labels.
    """
    # 1) Load the data so we can pick defaults
    df = pd.read_csv(sys.stdin) if input_file == Path("-") else pd.read_csv(input_file)

    # 2) Determine exactly three numeric features
    cols = numeric_cols or df.select_dtypes(include="number").columns[:3].tolist()
    if len(cols) != 3:
        raise typer.BadParameter("Must specify exactly three numeric columns for 3D.")

    # 3) Build a default output filename
    default_name = f"3d_cluster_{cols[0]}_{cols[1]}_{cols[2]}.png"
    if output_file is None:
        output_file = Path.cwd() / default_name
    output_file = _ensure_unique_path(output_file)

    # 4) Generate the plot
    fig = cluster_scatter_3d(
        df=df,
        numeric_cols=cols,
        cluster_col=cluster_col,
        scale=scale,
        output_path=output_file,
        save=save,
    )
    # tighten up marker and axes
    fig.update_traces(marker=dict(size=5, opacity=1))
    fig.update_layout(
        legend_title_text="Cluster",
        scene=dict(
            xaxis_title=cols[0],
            yaxis_title=cols[1],
            zaxis_title=cols[2],
        ),
    )

    # 5) Handle save vs display vs stdout
    if output_file == Path("-"):
        # stream png bytes
        fig.write_image(sys.stdout.buffer, format="png")
        logger.success("3D cluster scatter PNG written to stdout.")
    elif save:
        # write out to file
        fig.write_image(str(output_file))
        logger.success(f"3D cluster scatter saved to {output_file!r}")
    else:
        # open interactive plot
        fig.show(renderer="browser")
        logger.success("3D cluster scatter opened in browser (not saved).")


@app.command("cluster-subplot")
def batch_cluster_plot(
    input_file: Path = typer.Argument(..., help="Clustered CSV file, or '-' to read from stdin."),
    x_axis: Optional[str] = typer.Argument(..., help="Feature for X axis."),
    y_axis: Optional[str] = typer.Argument(..., help="Feature for Y axis."),
    cluster_prefix: str = typer.Option(
        "cluster_",
        "--cluster-col",
        "-cluster-col",
        help="Prefix for cluster columns in the DataFrame.",
    ),
    scale: float = typer.Option(1.0, "--scale", "-s", help="Multiplier for x and y axes."),
    save: bool = typer.Option(
        True, "--no-save", "-n", help="Save to file (default) or display only."
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Where to save the PNG; use '-' to write image bytes to stdout.",
    ),
):
    """
    Create a grid of 2D cluster-colored scatter plots for each column
    starting with `cluster_prefix`.
    """
    # 1) Load the data so we can pick defaults
    df = pd.read_csv(sys.stdin) if input_file == Path("-") else pd.read_csv(input_file)

    # 2) Infer x & y if omitted
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    if not numeric_columns:
        raise typer.BadParameter("No numeric columns found in the data.")
    x_col = x_axis or numeric_columns[0]
    y_col = y_axis or (numeric_columns[1] if len(numeric_columns) > 1 else numeric_columns[0])

    # 3) Find all cluster_* columns
    cluster_cols = sorted(
        [c for c in df.columns if c.startswith(cluster_prefix)],
        key=lambda c: int(c.replace(cluster_prefix, "")),
    )
    if not cluster_cols:
        raise typer.BadParameter(f"No columns found with prefix {cluster_prefix!r}")

    # 4) Build a default output filename
    default_name = f"{x_col}_vs_{y_col}_batch.png"
    if output_file is None:
        output_file = Path.cwd() / default_name
    # only uniquify when not streaming to stdout
    if output_file != Path("-"):
        output_file = _ensure_unique_path(output_file)

    # 5) Generate the plot (helper does the heavy lifting)
    fig = plot_batch_clusters(
        df=df,
        x_axis=x_col,
        y_axis=y_col,
        cluster_cols=cluster_cols,
        output_path=output_file,
        save=save,
        scale=scale,
    )

    # 6) Handle save vs display vs stdout
    if output_file == Path("-"):
        # stream PNG bytes to stdout
        fig.savefig(sys.stdout.buffer, format="png")
        logger.success("Cluster subplot PNG written to stdout.")
    elif save:
        # write to file
        fig.savefig(str(output_file))
        plt.close(fig)
        logger.success(f"Cluster subplot saved to {output_file!r}")
    else:
        # pop up interactive window
        plt.show()
        plt.close(fig)
        logger.success("Cluster subplot displayed (not saved).")


@app.command("biplot")
def plot_biplot(
    input_file: Path = typer.Argument(..., help="CSV file to read (use '-' to read from stdin)."),
    numeric_cols: List[str] = typer.Option(
        None,
        "--numeric-cols",
        "-numeric-cols",
        help="Defaults to all numeric columns if omitted.",
    ),
    skip_scores: bool = typer.Option(
        True,
        "--skip-scores",
        "-ss",
        help="Skip recomputing PC scores (assume df already has PC columns).",
    ),
    pc_x: int = typer.Option(0, "--pc-x", "-x", help="Index of PC for x-axis (0-based)."),
    pc_y: int = typer.Option(1, "--pc-y", "-y", help="Index of PC for y-axis (0-based)."),
    scale: float = typer.Option(1.0, "--scale", "-s", help="Arrow length multiplier."),
    hue_column: Optional[str] = typer.Argument(
        ..., help="Cluster column in DataFrame to color points by."
    ),
    save: bool = typer.Option(
        True, "--no-save", "-n", help="Save to file (default) or display only."
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output-file", "-o", help="Where to save PNG; use '-' for stdout."
    ),
):
    """
    2D Biplot: combines PC scores and loading vectors.
    """
    # 1) Load data
    df = pd.read_csv(sys.stdin) if input_file == Path("-") else pd.read_csv(input_file)

    # 2) Compute PCA summary
    summary = compute_pca_summary(
        df=df,
        numeric_cols=numeric_cols,
        hue_column=hue_column,
    )
    loadings = summary["loadings"]
    pve = summary["pve"]
    hue_ser = df[hue_column] if hue_column else None

    # 3) Default filename
    default_name = f"biplot_pc{pc_x + 1}-{pc_y + 1}.png"
    if output_file is None:
        output_file = Path.cwd() / default_name
    # if writing to disk
    if output_file != Path("-"):
        output_file = _ensure_unique_path(output_file)

    # 4) Draw
    fig = biplot(
        df=df,
        loadings=loadings,
        pve=pve,
        skip_scores=skip_scores,
        pc_x=pc_x,
        pc_y=pc_y,
        scale=scale,
        hue=hue_ser,
        save=save,
        output_path=output_file,
    )

    # 5) Handle save vs display vs stdout
    if output_file == Path("-"):
        logger.success("Biplot PNG written to stdout.")
        fig.savefig(sys.stdout.buffer, format="png")
        sys.stdout.buffer.flush()
    elif save:
        fig.savefig(str(output_file))
        plt.close(fig)
        logger.success(f"Biplot saved to {output_file!r}")
    else:
        plt.show()
        plt.close(fig)
        logger.success("Biplot displayed (not saved).")


@app.command("3d-biplot")
def plot_3d_biplot(
    input_file: Path = typer.Argument(..., help="CSV file or '-' for stdin."),
    numeric_cols: list[str] | None = typer.Argument(
        None,
        help="Exactly three features for PCA; if omitted, uses all numeric columns minus hue.",
    ),
    hue_column: str | None = typer.Option(
        None, "--hue-column", "-h", help="Column to color points by (e.g. cluster)."
    ),
    skip_scores: bool = typer.Option(
        True,
        "--skip-scores",
        "-ss",
        help="If set, skip recomputing PC scores (assume input already has PC columns).",
    ),
    pc_x: int = typer.Option(0, "--pc-x", "-x", help="PC for X axis (0-based)."),
    pc_y: int = typer.Option(1, "--pc-y", "-y", help="PC for Y axis (0-based)."),
    pc_z: int = typer.Option(2, "--pc-z", "-z", help="PC for Z axis (0-based)."),
    scale: float = typer.Option(1.0, "--scale", "-s", help="Loading arrow scale."),
    save: bool = typer.Option(
        True, "--no-save", "-n", help="Save PNG (default) or open interactive plot."
    ),
    output_file: Path | None = typer.Option(
        None, "--output-file", "-o", help="Where to save PNG; use '-' for stdout."
    ),
):
    """
    3D PCA biplot: sample scores + loading vectors in three dimensions.
    """
    default_name = "3d_biplot.png"

    # ─── 1) load DataFrame
    df = pd.read_csv(sys.stdin) if input_file == Path("-") else pd.read_csv(input_file)

    # ─── 2) pick numeric columns
    if numeric_cols is None:
        numerics = df.select_dtypes(include="number").columns.tolist()
        if hue_column in numerics:
            numerics.remove(hue_column)
    else:
        numerics = numeric_cols

    if skip_scores and len(numerics) < 3:
        raise typer.BadParameter("Need at least three numeric columns for PCA.")

    # ─── 3) compute PCA once
    summary = compute_pca_summary(df, numeric_cols=numerics, hue_column=hue_column)
    loadings = summary["loadings"]
    pve = summary["pve"]

    # ─── 4) finalize output path
    out_path = output_file or (Path.cwd() / default_name)
    out_path = _ensure_unique_path(out_path)

    # ─── 5) build under tqdm
    with tqdm(total=3, desc="3D Biplot", colour="green") as pbar:
        pbar.update(1)
        fig = biplot_3d(
            df=df,
            loadings=loadings,
            pve=pve,
            output_path=out_path,
            skip_scores=skip_scores,
            pc_x=pc_x,
            pc_y=pc_y,
            pc_z=pc_z,
            scale=scale,
            hue=(df[hue_column] if hue_column else None),
            save=save,
        )
        pbar.update(1)

        # ─── 6) save / stdout / show
        if out_path == Path("-"):
            fig.write_image(sys.stdout.buffer, format="png")
            logger.success("3D biplot PNG written to stdout.")
        elif save:
            fig.write_image(str(out_path))
            logger.success(f"3D biplot saved to {out_path!r}")
        else:
            fig.show(renderer="browser")
            logger.success("3D biplot opened in browser (not saved).")
        pbar.update(1)


if __name__ == "__main__":
    app()
