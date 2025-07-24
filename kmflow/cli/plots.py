from typing import Optional, List
import sys
import typer
from tqdm import tqdm
from pathlib import Path
from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil

import kmflow.utils.cli_utils as cli_utils
import kmflow.utils.plots_utils as plots_utils
import kmflow.utils.pca_utils as pca_utils

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
    plots_utils.plots_utils._run_plot_with_progress(
        name="Barplot",
        input_file=input_file,
        plot_fn=plots_utils.bar_plot,
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

    plots_utils.plots_utils._run_plot_with_progress(
        name="Histogram",
        input_file=input_file,
        plot_fn=plots_utils.histogram,
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

    plots_utils._run_plot_with_progress(
        name="Scatter plot",
        input_file=input_file,
        plot_fn=plots_utils.scatter_plot,
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

    plots_utils._run_plot_with_progress(
        name="Boxplot",
        input_file=input_file,
        plot_fn=plots_utils.box_plot,
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
    plots_utils._run_plot_with_progress(
        name="Violin",
        input_file=input_file,
        plot_fn=plots_utils.violin_plot,
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

    plots_utils._run_plot_with_progress(
        name="Correlation Heatmap",
        input_file=input_file,
        plot_fn=plots_utils.correlation_heatmap,
        kwargs={},
        output_file=output_file,
        default_name=default_name,
        save=save,
    )


@app.command("qq")
def qq_plt(
    input_file: Path = typer.Argument(..., help="CSV file path or '-' to read from stdin."),
    numeric_cols: str = typer.Option(
        "",
        "--numeric-col",
        "-nc",
        help="Comma-separated list of numeric columns for Q-Q plot, ex: 'Price, Quantity', (omit when using --all).",
        callback=lambda x: cli_utils.comma_split(x) if isinstance(x, str) else x,
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
            plots_utils._apply_cubehelix_style()
            n = len(cols)
            ncols = 3
            nrows = ceil(n / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
            axes_flat = axes.flatten()
            for i, col in enumerate(cols):
                plots_utils.qq_plot(df, col, output_path=None, save=False, ax=axes_flat[i])
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
                out = plots_utils._ensure_unique_path(output_file or Path.cwd() / default_name)
                fig.savefig(out)
                plt.close(fig)
                logger.success(f"Q-Q plots saved to {out!r}")
            pbar.update(1)

        return

    # ─── Single‑column (or sequential) mode ────────────────────────────────────────────────
    if not numeric_cols:
        raise typer.BadParameter("Specify --numeric-col (Shorthand: -nc) or use --all.")

    # new: read all of stdin into a DataFrame for reuse
    if input_file == Path("-"):
        df = pd.read_csv(sys.stdin)

    for col in numeric_cols:
        default_name = f"{col}_qq.png"

        if input_file == Path("-"):
            with tqdm(total=2, desc=f"Q-Q Plot: {col}", colour="green") as pbar:
                plots_utils._apply_cubehelix_style()
                fig, ax = plt.subplots()
                plots_utils.qq_plot(df, col, output_path=None, save=False, ax=ax)
                pbar.update(1)

                if output_file == Path("-"):
                    fig.savefig(sys.stdout.buffer, format="png")
                    logger.success(f"Q-Q Plot: {col} PNG written to stdout.")
                elif not save:
                    plt.show()
                    plt.close(fig)
                    logger.success(f"Q-Q Plot: {col} displayed (not saved).")
                else:
                    out = plots_utils._ensure_unique_path(output_file or Path.cwd() / default_name)
                    fig.savefig(out)
                    plt.close(fig)
                    logger.success(f"Q-Q Plot: {col} saved to {out!r}.")
                pbar.update(1)

        else:
            # existing file-based logic
            plots_utils._run_plot_with_progress(
                name=f"Q-Q Plot: {col}",
                input_file=input_file,
                plot_fn=plots_utils.qq_plot,
                kwargs={"numeric_col": col},
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

    plots_utils._run_plot_with_progress(
        name="Inertia",
        input_file=input_file,
        plot_fn=lambda df, output_path, save: plots_utils.inertia_plot(
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

    plots_utils._run_plot_with_progress(
        name="Silhouette",
        input_file=input_file,
        plot_fn=lambda df, output_path, save: plots_utils.silhouette_plot(
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

    plots_utils._run_plot_with_progress(
        name="Proportion of Variance (Scree)",
        input_file=input_file,
        plot_fn=lambda df, output_path, save: plots_utils.scree_plot(
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

    plots_utils._run_plot_with_progress(
        name="Proportion of Cumulative Variance",
        input_file=input_file,
        plot_fn=plots_utils.cumulative_var_plot,
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
    scale: float = typer.Option(1.0, "--scale", "-s", help="Multiplier for x/y axes."),
    save: bool = typer.Option(
        True, "--no-save", "-n", help="Save to file (default) or display only."
    ),
    output_file: Path = typer.Option(
        None, "--output-file", "-o", help="PNG path; use '-' for stdout."
    ),
):
    """
    Scatter plot of X vs. Y colored by cluster labels.
    """
    default_name = f"{x_axis}_vs_{y_axis}_cluster.png"
    out_path = output_file or (Path.cwd() / default_name)
    if out_path != Path("-"):
        out_path = plots_utils._ensure_unique_path(out_path)

    with tqdm(total=3, desc="Cluster Scatter", colour="green") as pbar:
        # 1) LOAD
        df = pd.read_csv(sys.stdin) if input_file == Path("-") else pd.read_csv(input_file)
        numeric = df.select_dtypes("number").columns.tolist()
        x_col = x_axis or numeric[0]
        y_col = y_axis or (numeric[1] if len(numeric) > 1 else numeric[0])
        pbar.update(1)

        # 2) PLOT
        ax = plots_utils.cluster_scatter(
            df=df,
            x_axis=x_col,
            y_axis=y_col,
            cluster_col=cluster_col,
            scale=scale,
            save=False,
            output_path=Path("unused.png"),
        )
        fig = ax.figure
        pbar.update(1)

        # 3) OUTPUT
        if out_path == Path("-"):
            fig.savefig(sys.stdout.buffer, format="png")
            logger.success("Cluster scatter PNG written to stdout.")
        elif save:
            fig.savefig(str(out_path))
            logger.success(f"Cluster scatter saved to {out_path!r}")
        else:
            plt.show()
            logger.success("Cluster scatter displayed (not saved).")
        pbar.update(1)


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
    default_name = "3d_cluster.png"
    out_path = output_file

    with tqdm(total=3, desc="3D Cluster Scatter", colour="green") as pbar:
        # ─── 1) LOAD + pick cols ───────────────────────────────────────────────
        df = pd.read_csv(sys.stdin) if input_file == Path("-") else pd.read_csv(input_file)
        cols = numeric_cols or df.select_dtypes(include="number").columns[:3].tolist()
        if len(cols) != 3:
            raise typer.BadParameter("Must specify exactly three numeric columns for 3D.")
        default_name = f"3d_cluster_{cols[0]}_{cols[1]}_{cols[2]}.png"

        # finalize out_path
        if out_path is None:
            out_path = Path.cwd() / default_name
        if out_path != Path("-"):
            out_path = plots_utils._ensure_unique_path(out_path)
        pbar.update(1)

        # ─── 2) PLOT ───────────────────────────────────────────────────────────
        fig = plots_utils.cluster_scatter_3d(
            df=df,
            numeric_cols=cols,
            cluster_col=cluster_col,
            scale=scale,
            output_path=out_path,
            save=False,
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
        pbar.update(1)

        # ─── 3) OUTPUT ─────────────────────────────────────────────────────────
        if out_path == Path("-"):
            fig.write_image(sys.stdout.buffer, format="png")
            logger.success("3D cluster scatter PNG written to stdout.")
        elif save:
            fig.write_image(str(out_path))
            logger.success(f"3D cluster scatter saved to {out_path!r}")
        else:
            fig.show(renderer="browser")
            logger.success("3D cluster scatter opened in browser (not saved).")
        pbar.update(1)


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
    default_name = "cluster_subplot.png"
    out_path = output_file

    with tqdm(total=3, desc="Cluster Subplot", colour="green") as pbar:
        # ─── 1) LOAD + infer defaults ───────────────────────────────
        df = pd.read_csv(sys.stdin) if input_file == Path("-") else pd.read_csv(input_file)

        numeric_columns = df.select_dtypes(include="number").columns.tolist()
        if not numeric_columns:
            raise typer.BadParameter("No numeric columns found in the data.")

        x_col = x_axis or numeric_columns[0]
        y_col = y_axis or (numeric_columns[1] if len(numeric_columns) > 1 else numeric_columns[0])

        cluster_cols = sorted(
            [c for c in df.columns if c.startswith(cluster_prefix)],
            key=lambda c: int(c.replace(cluster_prefix, "")),
        )
        if not cluster_cols:
            raise typer.BadParameter(f"No columns found with prefix {cluster_prefix!r}")

        default_name = f"{x_col}_vs_{y_col}_batch.png"
        if out_path is None:
            out_path = Path.cwd() / default_name
        if out_path != Path("-"):
            out_path = plots_utils._ensure_unique_path(out_path)

        pbar.update(1)

        # ─── 2) DRAW (helper returns a Matplotlib Figure) ───────────
        fig = plots_utils.plot_batch_clusters(
            df=df,
            x_axis=x_col,
            y_axis=y_col,
            cluster_cols=cluster_cols,
            output_path=out_path,
            save=False,
            scale=scale,
        )
        pbar.update(1)

        # ─── 3) OUTPUT ───────────────────────────────────────────────
        if out_path == Path("-"):
            # stream PNG to stdout
            fig.savefig(sys.stdout.buffer, format="png")
            logger.success("Cluster subplot PNG written to stdout.")
        elif save:
            # write to disk
            fig.savefig(str(out_path))
            logger.success(f"Cluster subplot saved to {out_path!r}")
        else:
            # interactive display
            plt.show()
            logger.success("Cluster subplot displayed (not saved).")

        pbar.update(1)


@app.command("biplot")
def plot_biplot(
    input_file: Path = typer.Argument(..., help="CSV file to read (use '-' to read from stdin)."),
    numeric_cols: list[str] = typer.Option(
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
    hue_column: str = typer.Argument(..., help="Cluster column in DataFrame to color points by."),
    save: bool = typer.Option(
        True, "--no-save", "-n", help="Save to file (default) or display only."
    ),
    output_file: Path = typer.Option(
        None, "--output-file", "-o", help="Where to save PNG; use '-' for stdout."
    ),
):
    """
    2D Biplot: combines PC scores and loading vectors.
    """
    # ─── decide output path ───────────────────────────────────────
    default_name = f"biplot_pc{pc_x + 1}-{pc_y + 1}.png"
    out_path = output_file or (Path.cwd() / default_name)
    if out_path != Path("-"):
        out_path = plots_utils._ensure_unique_path(out_path)

    with tqdm(total=3, desc="Biplot", colour="green") as pbar:
        # 1) LOAD ────────────────────────────────────────────────
        df = pd.read_csv(sys.stdin) if input_file == Path("-") else pd.read_csv(input_file)
        pbar.update(1)

        # 2) DRAW ────────────────────────────────────────────────
        summary = pca_utils.compute_pca(
            df=df,
            numeric_cols=numeric_cols,
            hue_column=hue_column,
        )
        loadings = summary["loadings"]
        pve = summary["pve"]
        hue_ser = df[hue_column] if hue_column else None

        fig = plots_utils.biplot(
            df=df,
            loadings=loadings,
            pve=pve,
            skip_scores=skip_scores,
            pc_x=pc_x,
            pc_y=pc_y,
            scale=scale,
            hue=hue_ser,
            save=False,
            output_path=Path("unused.png"),
        )
        pbar.update(1)

        # 3) OUTPUT ──────────────────────────────────────────────
        if out_path == Path("-"):
            fig.savefig(sys.stdout.buffer, format="png")
            logger.success("Biplot PNG written to stdout.")
        elif save:
            fig.savefig(str(out_path))
            plt.close(fig)
            logger.success(f"Biplot saved to {out_path!r}")
        else:
            plt.show()
            plt.close(fig)
            logger.success("Biplot displayed (not saved).")
        pbar.update(1)


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

    # ─── 1) load DataFrame ─────────────────────────────────────
    df = pd.read_csv(sys.stdin) if input_file == Path("-") else pd.read_csv(input_file)

    # ─── 2) pick numeric columns ──────────────────────────────
    if numeric_cols is None:
        numerics = df.select_dtypes(include="number").columns.tolist()
        if hue_column in numerics:
            numerics.remove(hue_column)
    else:
        numerics = numeric_cols

    if not skip_scores and len(numerics) < 3:
        raise typer.BadParameter("Need at least three numeric columns for PCA.")

    # ─── 3) compute PCA once ───────────────────────────────────
    summary = pca_utils.compute_pca(df, numeric_cols=numerics, hue_column=hue_column)
    loadings = summary["loadings"]
    pve = summary["pve"]

    # ─── 4) finalize output path ──────────────────────────────
    out_path = output_file or (Path.cwd() / default_name)
    if out_path != Path("-"):
        out_path = plots_utils._ensure_unique_path(out_path)

    # ─── 5) build & report under tqdm ────────────────────────
    with tqdm(total=3, desc="3D Biplot", colour="green") as pbar:
        # step 1: prepare figure
        fig = plots_utils.biplot_3d(
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

        # step 2: annotate / tighten
        pbar.update(1)

        # step 3: output
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
