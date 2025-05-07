from typing import Optional
import typer
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import pandas as pd

from tennis_racquet_analysis.config import DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR
from tennis_racquet_analysis.preprocessing_utils import load_data
from tennis_racquet_analysis.plots_utils import (
    _save_fig,
    df_to_array,
    df_to_labels,
    compute_linkage,
    dendrogram_plot,
    histogram,
    scatter_plot,
    box_plot,
    violin_plot,
    correlation_matrix_heatmap,
    qq_plot,
    qq_plots_all,
    inertia_plot,
    silhouette_plot,
)

app = typer.Typer()


@app.command("hist")
def hist(
    input_file: str = typer.Argument("csv filename."),
    dir_label: str = typer.Argument("Sub-folder under data/"),
    x_axis: str = typer.Argument(..., help="Column to histogram."),
    num_bins: int = typer.Option(10, "--bins", "-b", help="Number of bins."),
    output_dir: Path = typer.Option(
        FIGURES_DIR,
        "--output-dir",
        "-o",
        dir_okay=True,
        file_okay=False,
        help="Where to save the .png plot.",
    ),
    no_save: bool = typer.Option(
        False,
        "--no-save",
        "-n",
        help="Generate plot, but don't write to disk.",
    ),
):
    input_path = DATA_DIR / dir_label / input_file
    df = load_data(input_path)
    output_path = output_dir / f"{Path(input_file).stem}_{x_axis}_hist.png"
    steps = tqdm(total=1, desc="Histogram", ncols=100)
    histogram(df, x_axis, num_bins, output_path, save=not no_save)
    steps.update(1)
    steps.close()
    if not no_save:
        logger.success(f"Histogram saved to {output_path}")
    else:
        logger.success("Histogram generated (not saved to disk).")


@app.command("scatter")
def scatter(
    input_file: str = typer.Argument(..., help="csv filename."),
    dir_label: str = typer.Argument("Sub-folder under data/"),
    x_axis: str = typer.Argument(..., help="X-axis column."),
    y_axis: str = typer.Argument(..., help="Y-axis column."),
    output_dir: Path = typer.Option(
        FIGURES_DIR,
        "--output-dir",
        "-o",
        dir_okay=True,
        file_okay=False,
    ),
    no_save: bool = typer.Option(
        False, "--no-save", "-n", help="Generate plot, but don't write to disk."
    ),
):
    input_path = DATA_DIR / dir_label / input_file
    df = load_data(input_path)
    output_path = output_dir / f"{Path(input_file).stem}_{x_axis}_vs._{y_axis}_scatter.png"
    steps = tqdm(total=1, desc="Scatter", ncols=100)
    scatter_plot(df, x_axis, y_axis, output_path, save=not no_save)
    steps.update(1)
    steps.close()
    if not no_save:
        logger.success(f"Scatter plot saved to {output_path}")
    else:
        logger.success("Scatter plot generated (not saved to disk).")


@app.command("boxplot")
def boxplt(
    input_file: str = typer.Argument(..., help="csv filename."),
    dir_label: str = typer.Argument("Sub-folder under data/"),
    y_axis: str = typer.Argument(..., help="Y-axis column."),
    brand: str = typer.Option(
        None,
        "--brand",
        "-b",
        help="Filter to a single brand (defaults to all).",
    ),
    orient: str = typer.Option("v", "--orient", "-a", help="Orientation of the plot."),
    output_dir: Path = typer.Option(
        FIGURES_DIR,
        "--output-dir",
        "-o",
        dir_okay=True,
        file_okay=False,
    ),
    no_save: bool = typer.Option(
        False, "--no-save", "-n", help="Generate plot, but don't write to disk."
    ),
):
    input_path = DATA_DIR / dir_label / input_file
    df = load_data(input_path)
    stem = Path(input_file).stem
    stem_label = brand.lower() if brand else "by_brand"
    file_name = f"{stem}_{stem_label}_{y_axis}_boxplot.png"
    output_path = output_dir / file_name
    steps = tqdm(total=1, desc="Boxplot", ncols=100)
    box_plot(
        df=df,
        y_axis=y_axis,
        output_path=output_path,
        brand=brand,
        orient=orient,
        save=not no_save,
    )
    steps.update(1)
    steps.close()
    if no_save:
        logger.success("Box plot generated (not saved to disk).")
    else:
        logger.success(f"Box plot saved to {output_path!r}")


@app.command("violin")
def violinplt(
    input_file: str = typer.Argument(..., help="csv filename."),
    dir_label: str = typer.Argument("Sub-folder under data/"),
    y_axis: str = typer.Argument(..., help="Y-axis column."),
    brand: str = typer.Option(
        None,
        "--brand",
        "-b",
        help="Filter to a single brand (defaults to all).",
    ),
    orient: str = typer.Option("v", "--orient", "-a", help="Orientation of the plot."),
    inner: str = typer.Option(
        "box",
        "--inner",
        "-i",
        help="Representation of the data in the interior of the violin plot. "
        "Use 'box', 'point', 'quartile', 'point', or 'stick' inside the violin.",
    ),
    output_dir: Path = typer.Option(
        FIGURES_DIR,
        "--output-dir",
        "-o",
        dir_okay=True,
        file_okay=False,
    ),
    no_save: bool = typer.Option(
        False, "--no-save", "-n", help="Generate plot, but don't write to disk."
    ),
):
    input_path = DATA_DIR / dir_label / input_file
    df = load_data(input_path)
    stem = Path(input_file).stem
    stem_label = brand.lower() if brand else "by_brand"
    file_name = f"{stem}_{stem_label}_{y_axis}_violin.png"
    output_path = output_dir / file_name
    steps = tqdm(total=1, desc="Violin", ncols=100)
    violin_plot(
        df=df,
        y_axis=y_axis,
        output_path=output_path,
        brand=brand,
        orient=orient,
        inner=inner,
        save=not no_save,
    )
    steps.update(1)
    steps.close()
    if no_save:
        logger.success("Violin plot generated (not saved to disk).")
    else:
        logger.success(f"Violin plot saved to {output_path!r}")


@app.command("dendrogram")
def dendrogram_plt(
    input_file: Path = typer.Argument(..., help="csv filename."),
    dir_label: str = typer.Argument("Sub-folder under data/"),
    label_col: Optional[str] = typer.Option(
        None,
        "--label",
        "-l",
        help="Column to use for leaf labels; if omitted leaves are numbered by index.",
    ),
    linkage_method: str = typer.Option(
        "centroid",
        "--method",
        "-m",
        help="Methods for calculating the distance between clusters are: 'single', 'complete', 'average', 'weighted', 'centroid' (default), 'median', and 'ward'.",
    ),
    distance_metric: str = typer.Option(
        "euclidean",
        "--metric",
        "-d",
        help="Pairwise distances between observations in n-dimensional space. The different distance metrics that are available to use are:"
        "'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean' (default), 'hamming', 'jaccard', "
        "'jensenshannon', 'kulczynski1', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',"
        "'sokalsneath', 'sqeuclidean', 'yule'.",
    ),
    ordering: bool = typer.Option(
        True,
        "--ordering",
        "-ord",
        help="Optimal ordering set to 'True' by default, which results in more intuitive tree structure when the data are visualized."
        "However, the algorithm can be slow especially with larger datasets.",
    ),
    output_path: Path = typer.Option(
        FIGURES_DIR,
        "--output-path",
        "-o",
        dir_okay=True,
        file_okay=False,
    ),
    orient: str = typer.Option(
        "right",
        "--orient",
        "-ort",
        help="Direction to plot the dendrogram. The following directions are: 'top', 'bottom', 'left', and 'right' (default).",
    ),
    no_save: bool = typer.Option(
        False,
        "--no-save",
        "-n",
        help="Generate plot, but don't write to disk.",
    ),
):
    input_path = DATA_DIR / dir_label / input_file
    df = load_data(input_path)
    stem = Path(input_file).stem
    file_name = f"{stem}_{label_col}_{linkage_method}_{distance_metric}_dendrogram.png"
    output_path = output_path / file_name
    array = df_to_array(df)
    if label_col is None:
        labels = None
    else:
        if label_col not in df.columns:
            raise typer.BadParameter(
                f"`{label_col}` is not a column in your data. Available: {list(df.columns)}"
            )
        labels = df_to_labels(df, label_col)
    Z = compute_linkage(
        array, method=linkage_method, metric=distance_metric, optimal_ordering=ordering
    )
    steps = tqdm(total=1, desc="Dendrogram", ncols=100)
    dendrogram_plot(
        Z=Z,
        labels=labels,
        output_path=output_path,
        orient=orient,
        save=not no_save,
    )
    steps.update(1)
    steps.close()
    logger.success(f"Dendrogram {'generated' if no_save else 'saved to'} {output_path}")


@app.command("heatmap")
def corr_heat(
    input_file: str = typer.Argument("csv filename."),
    dir_label: str = typer.Argument("Sub-folder under data/"),
    output_dir: Path = typer.Option(
        FIGURES_DIR,
        "--output-dir",
        "-o",
        dir_okay=True,
        file_okay=False,
        help="Where to save the .png plot.",
    ),
    no_save: bool = typer.Option(
        False,
        "--no-save",
        "-n",
        help="Generate plot, but don't write to disk.",
    ),
):
    input_path = DATA_DIR / dir_label / input_file
    df = load_data(input_path)
    output_path = output_dir / f"{Path(input_file).stem}_heatmap.png"
    steps = tqdm(total=1, desc="Heatmap", ncols=100)
    correlation_matrix_heatmap(df, output_path, save=not no_save)
    steps.update(1)
    steps.close()
    if not no_save:
        logger.success(f"Heatmap saved to {output_path}")
    else:
        logger.success("Heatmap generated (not saved to disk).")


@app.command("qq")
def qq(
    input_file: str = typer.Argument(..., help="csv filename under the data subfolder."),
    dir_label: str = typer.Argument(..., help="Sub-folder under data/"),
    column: list[str] = typer.Option(
        [], "--column", "-c", help="Column(s) to plot; repeat for multiple."
    ),
    all_cols: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Plot Q-Q for all numeric columns.",
    ),
    output_dir: Path = typer.Option(
        FIGURES_DIR,
        "--output-dir",
        "-o",
        help="Where to save plot(s).",
    ),
    no_save: bool = typer.Option(
        False,
        "--no-save",
        "-n",
        help="Generate plots, but don't write to disk.",
    ),
):
    df = load_data(DATA_DIR / dir_label / input_file)
    if column and not all_cols:
        for col in column:
            stem = Path(input_file).stem
            file_name = f"{stem}_{col}_qq.png"
            output_path = output_dir / file_name
            qq_plot(df=df, column=col, output_path=output_path, save=not no_save)
            if not no_save:
                logger.success(f"Saved Q-Q Plot for {col.capitalize()} to {output_path!r}")
    elif all_cols:
        stem = Path(input_file).stem
        fig = qq_plots_all(df=df, output_dir=output_dir, columns=None, ncols=3, save=False)
        fig.suptitle(f"Q-Q Plots: {stem} Data")
        fig.tight_layout()
        file_name = f"{stem}_qq_plots_all.png"
        output_path = output_dir / file_name
        if not no_save:
            _save_fig(fig, output_path)
            logger.success(f"Saved combined Q-Q plots to {output_path!r}")
        else:
            fig.show()
    else:
        raise typer.BadParameter("Specify one or more --column or use --all")


@app.command("elbow")
def elbow_plot(
    input_file: str = typer.Argument(..., help="csv from `inertia` command."),
    dir_label: str = typer.Argument(..., help="Sub-folder under processed data/"),
    output_dir: Path = typer.Option(
        FIGURES_DIR,
        "--output-dir",
        "-o",
        dir_okay=True,
        file_okay=False,
        help="Where to save the elbow plot png.",
    ),
    no_save: bool = typer.Option(
        False,
        "--no-save",
        "-n",
        help="Show plot, but don't save.",
    ),
):
    df = load_data(DATA_DIR / dir_label / input_file)
    stem = Path(input_file).stem
    output_path = output_dir / f"{stem}_elbow.png"
    fig = inertia_plot(
        df,
        output_path,
        save=no_save,
    )
    if not no_save:
        _save_fig(fig, output_path)
        logger.success(f"Elbow Plot saved to {output_path!r}")
    else:
        fig.show()


@app.command("silhouette")
def plot_silhouette(
    input_file: str = typer.Argument(..., help="CSV from `silhouette` command."),
    dir_label: str = typer.Argument(..., help="Sub-folder under processed data/"),
    output_dir: Path = typer.Option(
        FIGURES_DIR,
        "--output-dir",
        "-o",
        dir_okay=True,
        file_okay=False,
        help="Where to save the silhouette plot PNG.",
    ),
    no_save: bool = typer.Option(False, "--no-save", "-n", help="Show plot but don’t save."),
):
    df = load_data(DATA_DIR / dir_label / input_file)
    stem = Path(input_file).stem
    output_path = output_dir / f"{stem}_silhouette.png"
    fig = silhouette_plot(df, output_path, save=not no_save)
    if not no_save:
        _save_fig(fig, output_path)
        logger.success(f"Silhouette Plot saved to {output_path!r}")
    else:
        fig.show()


if __name__ == "__main__":
    app()
