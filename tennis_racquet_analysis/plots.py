from typing import Optional
import typer
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import ceil
import plotly.express as px


from tennis_racquet_analysis.config import DATA_DIR, FIGURES_DIR, PROCESSED_DATA_DIR
from tennis_racquet_analysis.preprocessing_utils import load_data
from tennis_racquet_analysis.plots_utils import (
    _save_fig,
    _apply_cubehelix_style,
    histogram,
    scatter_plot,
    box_plot,
    violin_plot,
    correlation_matrix_heatmap,
    qq_plot,
    inertia_plot,
    silhouette_plot,
    scree_plot,
    cumulative_prop_var_plot,
    pca_biplot,
    pca_biplot_3d,
    cluster_scatter,
    cluster_scatter_3d,
    plot_batch_clusters,
)

from tennis_racquet_analysis.preprocessing_utils import compute_pca_summary

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
    stem = Path(input_file).stem
    if column and not all_cols:
        for col in tqdm(column, desc="Q-Q Plot"):
            file_name = f"{stem}_{col}_qq.png"
            output_path = output_dir / file_name
            qq_plot(df=df, column=col, output_path=output_path, save=not no_save)
            if not no_save:
                logger.success(f"Saved Q-Q Plot for '{col}' -> {output_path!r}")
    elif all_cols:
        cols = df.select_dtypes(include="number").columns.tolist()
        n = len(cols)
        ncols = 3
        nrows = ceil(n / ncols)
        _apply_cubehelix_style()
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))
        axes_flat = axes.flatten()
        for i, col in enumerate(tqdm(cols, desc="Q-Q Plots")):
            ax = axes_flat[i]
            qq_plot(df=df, column=col, output_path=None, save=False, ax=ax)
            ax.set_title(col.capitalize())
        for ax in axes_flat[n:]:
            ax.set_visible(False)
        fig.suptitle(f"Q-Q Plots: {stem} Data")
        fig.tight_layout()
        file_name = f"{stem}_qq_all.png"
        output_path = output_dir / file_name
        if not no_save:
            _save_fig(fig, output_path)
            logger.success(f"Saved Combined Q-Q Plots -> {output_path!r}")
        else:
            fig.show()
    else:
        raise typer.BadParameter("Specify one or more --column or use --all.")


@app.command("inertia")
def elbow_plot(
    input_file: str = typer.Argument(..., help="csv from `inertia` command."),
    input_dir: str = typer.Option(
        "processed",
        "--input-dir",
        "-d",
        help="Sub-folder under data/ (e.g. external, interim, processed, raw), where the input file lives.",
    ),
    output_dir: Path = typer.Option(
        FIGURES_DIR,
        "--output-dir",
        "-o",
        dir_okay=True,
        file_okay=False,
        help="Save the inertia plot PNG to the 'figures' directory.",
    ),
    no_save: bool = typer.Option(False, "--no-save", "-n", help="Show plot, but don't save."),
):
    df = load_data(DATA_DIR / input_dir / input_file)
    stem = Path(input_file).stem
    output_path = output_dir / f"{stem}.png"
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
    input_dir: str = typer.Option(
        "processed",
        "--input-dir",
        "-d",
        help="Sub-folder under data/ (e.g. external, interim, processed, raw), where the input file lives.",
    ),
    output_dir: Path = typer.Option(
        FIGURES_DIR,
        "--output-dir",
        "-o",
        dir_okay=True,
        file_okay=False,
        help="Save the silhouette plot PNG to the 'figures' directory.",
    ),
    no_save: bool = typer.Option(False, "--no-save", "-n", help="Show plot but don’t save."),
):
    df = load_data(DATA_DIR / input_dir / input_file)
    stem = Path(input_file).stem
    output_path = output_dir / f"{stem}.png"
    fig = silhouette_plot(df, output_path, save=not no_save)
    if not no_save:
        _save_fig(fig, output_path)
        logger.success(f"Silhouette Plot saved to {output_path!r}")
    else:
        fig.show()


@app.command("cluster")
def cluster_plot(
    input_file: str = typer.Argument(..., help="csv filename under data subfolder."),
    input_dir: str = typer.Option(
        "processed",
        "--input-dir",
        "-d",
        help="Sub-folder under data/ (e.g. external, interim, processed, raw), where the input file lives.",
    ),
    x_axis: Optional[str] = typer.Option(
        None,
        "--x-axis",
        "-x",
        help="Feature for X axis.",
    ),
    y_axis: Optional[str] = typer.Option(None, "--y-axis", "-y", help="Feature for Y axis."),
    label_column: str = typer.Option(
        "cluster_",
        "--label-column",
        "-l",
        help="Column with cluster labels.",
    ),
    output_dir: Path = typer.Option(
        FIGURES_DIR,
        "--output-dir",
        "-o",
        help="Where to save the plot.",
    ),
    no_save: bool = typer.Option(
        False,
        "--no-save",
        "-n",
        help="Don't write to disk.",
    ),
):
    df = load_data(DATA_DIR / input_dir / input_file)
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    x_col = x_axis or numeric_columns[0]
    y_col = y_axis or (numeric_columns[1] if len(numeric_columns) > 1 else numeric_columns[0])
    with tqdm(total=1, desc="Generating Cluster Scatter") as pbar:
        output_path = output_dir / f"{Path(input_file).stem}_{x_col}_vs_{y_col}_cluster.png"
        (
            cluster_scatter(
                df=df,
                x_axis=x_col,
                y_axis=y_col,
                label_column=label_column,
                output_path=output_path,
                save=not no_save,
            ),
        )
        pbar.update(1)
    if not no_save:
        logger.success(f"Cluster Scatter saved to {output_path!r}")
    else:
        logger.success("Cluster Scatter generated (not saved to disk).")


@app.command("cluster-3d")
def cluster_3d_plot(
    input_file: str = typer.Argument(..., help="Clustered csv filename"),
    input_dir: str = typer.Option(
        "processed",
        "--input-dir",
        "-d",
        help="Sub-folder under data/ (e.g. external, interim, processed, raw), where the input file lives.",
    ),
    features: list[str] = typer.Option(
        None,
        "--feature",
        "-f",
        help="Exactly three numeric columns to plot; defaults to first three.",
    ),
    label_column: str = typer.Option(
        "cluster_",
        "--label-column",
        "-l",
        help="Name of the cluster label column (e.g. cluster_5).",
    ),
    output_dir: Path = typer.Option(
        FIGURES_DIR,
        "--output-dir",
        "-o",
        dir_okay=True,
        file_okay=False,
    ),
    no_save: bool = typer.Option(
        False,
        "--no-save",
        "-n",
        help="Don't write to disk. Opens html plot in browser.",
    ),
):
    with tqdm(total=3, desc="Cluster-3D") as progress_bar:
        df = load_data(DATA_DIR / input_dir / input_file)
        progress_bar.update(1)
        num_cols = df.select_dtypes(include="number").columns.tolist()
        chosen = features or num_cols[:3]
        if len(chosen) != 3:
            raise typer.BadParameter("Must specify exactly three features for 3D.")
        df[label_column] = df[label_column].astype(str)
        cluster_order = sorted(df[label_column].unique(), key=lambda x: int(x))
        stem = Path(input_file).stem
        base = f"{stem}_{label_column}_3d"
        fig = px.scatter_3d(
            df,
            x=chosen[0],
            y=chosen[1],
            z=chosen[2],
            color=label_column,
            category_orders={label_column: cluster_order},
            title=f"3D Cluster Scatter (k={label_column.split('_')[-1]})",
        )
        fig.update_traces(marker=dict(size=5, opacity=1))
        fig.update_layout(
            legend_title_text="Cluster",
            scene=dict(
                xaxis_title=chosen[0],
                yaxis_title=chosen[1],
                zaxis_title=chosen[2],
            ),
        )
        progress_bar.update(1)
        if not no_save:
            png_path = output_dir / f"{base}.png"
            _ = cluster_scatter_3d(
                df=df,
                features=chosen,
                label_column=label_column,
                output_path=png_path,
            )
            logger.success(f"Static PNG saved to {png_path!r}")
            progress_bar.update(1)
        else:
            fig.show()
            progress_bar.update(1)


@app.command("cluster-subplot")
def batch_cluster_plot(
    input_file: str = typer.Argument(..., help="csv filename under data subfolder"),
    input_dir: str = typer.Option(
        "processed",
        "--input-dir",
        "-d",
        help="Sub-folder under data/ (e.g. external, interim, processed, raw), where the input file lives.",
    ),
    x_axis: Optional[str] = typer.Option(
        None,
        "--x-axis",
        "-x",
        help="Feature for X axis (defaults to first numeric column).",
    ),
    y_axis: Optional[str] = typer.Option(
        None,
        "--y-axis",
        "-y",
        help="Feature for Y axis (defaults to second numeric column).",
    ),
    label_column: str = typer.Option(
        "cluster_",
        "--label",
        "-l",
        help="Name of the column containing clusters in DataFrame from `input_file`",
    ),
    output_dir: Path = typer.Option(
        FIGURES_DIR,
        "--output-dir",
        "-o",
        dir_okay=True,
        file_okay=False,
        help="Where to save the batch-cluster plot.",
    ),
):
    df = load_data(DATA_DIR / input_dir / input_file)
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    if not numeric_columns:
        raise typer.BadParameter("No numeric columns found in your data.")
    x_col = x_axis or numeric_columns[0]
    y_col = y_axis or (numeric_columns[1] if len(numeric_columns) > 1 else numeric_columns[0])
    cluster_columns = sorted(
        (c for c in df.columns if c.startswith(label_column)),
        key=lambda c: int(c.replace(label_column, "")),
    )
    if not cluster_columns:
        raise typer.BadParameter(f"No columns found with prefix {label_column!r}")
    output_path = output_dir / f"{Path(input_file).stem}_{x_col}_vs_{y_col}_batch.png"
    with tqdm(total=1, desc="Generating Batch Subplots") as progress_bar:
        plot_batch_clusters(
            df,
            x_axis=x_col,
            y_axis=y_col,
            cluster_columns=cluster_columns,
            output_path=output_path,
            save=True,
        )
        progress_bar.update(1)
    logger.success(f"Saved batch-cluster plot for {cluster_columns} -> {output_path!r}")


@app.command("scree")
def plot_scree(
    input_file: str = typer.Argument(..., help="csv file."),
    input_dir: Path = typer.Option(
        PROCESSED_DATA_DIR,
        "--input_dir",
        "-d",
        exists=True,
        dir_okay=True,
        file_okay=True,
        help="Directory where feature-engineered files live.",
    ),
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
    df = load_data(DATA_DIR / input_dir / input_file)
    stem = Path(input_file).stem
    output_path = output_dir / f"{stem}_scree.png"
    fig = scree_plot(
        df,
        output_path,
        save=no_save,
    )
    if not no_save:
        _save_fig(fig, output_path)
        logger.success(f"Scree Plot saved to {output_path!r}")
    else:
        fig.show()


@app.command("cumulative-prop-var")
def plot_cumulative_prop_var(
    input_file: str = typer.Argument(..., help="csv file."),
    input_dir: Path = typer.Option(
        PROCESSED_DATA_DIR,
        "--input_dir",
        "-d",
        exists=True,
        dir_okay=True,
        file_okay=True,
        help="Directory where feature-engineered files live.",
    ),
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
    df = load_data(DATA_DIR / input_dir / input_file)
    stem = Path(input_file).stem
    output_path = output_dir / f"{stem}_cumulative_prop_var.png"
    fig = cumulative_prop_var_plot(
        df,
        output_path,
        save=no_save,
    )
    if not no_save:
        _save_fig(fig, output_path)
        logger.success(f"Cumulative Prop. Variance Plot saved to {output_path!r}")
    else:
        fig.show()


@app.command("pca-biplot")
def plot_pca_biplot(
    input_file: str = typer.Argument(..., help="CSV filename under data subfolder."),
    input_dir: Path = typer.Option(
        PROCESSED_DATA_DIR,
        "-d",
        "--input-dir",
        exists=True,
        dir_okay=True,
        file_okay=True,
        help="Directory where scaled data files live.",
    ),
    feature_columns: list[str] = typer.Option(
        None,
        "-f",
        "--feature-column",
        help="Numeric column(s) to include; repeat flag to add more. Defaults to all.",
    ),
    random_state: int = typer.Option(4572, "-s", "--seed", help="Random seed for PCA."),
    pc_x: int = typer.Option(0, "--pc-x", help="Principal component for x-axis (0-indexed)."),
    pc_y: int = typer.Option(1, "--pc-y", help="Principal component for y-axis (0-indexed)."),
    scale: float = typer.Option(1.0, "--scale", help="Arrow length multiplier for loadings."),
    figsize: tuple[float, float] = typer.Option(
        (20, 14), "--figsize", help="Figure size (width height)."
    ),
    hue_column: Optional[str] = typer.Option(
        None,
        "--hue",
        help="Column name for coloring samples (Will be excluded from PCA summary helper).",
    ),
    output_dir: Path = typer.Option(
        FIGURES_DIR,
        "-o",
        "--output-dir",
        exists=True,
        dir_okay=True,
        file_okay=False,
        help="Directory to save the biplot PNG.",
    ),
):
    df = load_data(input_dir / input_file)

    summary = compute_pca_summary(
        df=df, feature_columns=feature_columns, hue_column=hue_column, random_state=random_state
    )
    loadings = summary["loadings"]
    pve = summary["pve"]

    hue = df[hue_column] if hue_column else None

    fig = pca_biplot(
        df=df,
        loadings=loadings,
        pve=pve,
        pc_x=pc_x,
        pc_y=pc_y,
        scale=scale,
        figsize=figsize,
        hue=hue,
        save=False,
        output_path=None,
    )

    stem = Path(input_file).stem
    out_file = f"{stem}_pca_biplot_PC{pc_x + 1}_{pc_y + 1}.png"
    out_path = output_dir / out_file
    _save_fig(fig, out_path)
    logger.success(f"Saved PCA biplot → {out_path!r}")


@app.command("pca-biplot-3d")
def plot_3d_pca_biplot(
    input_file: str = typer.Argument(..., help="CSV filename under data subfolder."),
    input_dir: Path = typer.Option(
        PROCESSED_DATA_DIR,
        "-d",
        "--input-dir",
        exists=True,
        dir_okay=True,
        file_okay=True,
        help="Directory of csv.",
    ),
    feature_columns: list[str] = typer.Option(
        None,
        "-f",
        "--feature-column",
        help="Numeric column(s) to include; repeat flag to add more. Defaults to all.",
    ),
    random_state: int = typer.Option(4572, "-s", "--seed", help="Random seed for PCA."),
    pc_x: int = typer.Option(0, "--x", help="Principal component for x-axis (0-indexed)."),
    pc_y: int = typer.Option(1, "--y", help="Principal component for y-axis (0-indexed)."),
    pc_z: int = typer.Option(2, "--z", help="Principal component for z-axis (0-indexed)."),
    scale: float = typer.Option(1.0, "--scale", help="Arrow length multiplier for loadings."),
    hue_column: Optional[str] = typer.Option(
        None,
        "--hue",
        help="Column name for coloring samples (Will be excluded from PCA summary helper).",
    ),
    output_dir: Path = typer.Option(
        FIGURES_DIR,
        "-o",
        "--output-dir",
        exists=True,
        dir_okay=True,
        file_okay=False,
        help="Directory to save the biplot PNG.",
    ),
    no_save: bool = typer.Option(
        False,
        "--no-save",
        "-n",
        help="Show plot, but don't save.",
    ),
):
    df = load_data(input_dir / input_file)

    summary = compute_pca_summary(
        df=df, feature_columns=feature_columns, hue_column=hue_column, random_state=random_state
    )
    loadings, pve = summary["loadings"], summary["pve"]
    hue = df[hue_column] if hue_column else None

    stem = Path(input_file).stem
    png_path = output_dir / f"{stem}_3d_pca_biplot.png"

    pca_biplot_3d(
        df=df,
        loadings=loadings,
        pve=pve,
        pc_x=pc_x,
        pc_y=pc_y,
        pc_z=pc_z,
        scale=scale,
        hue=hue,
        output_path=None if no_save else png_path,
        show=no_save,
    )

    if not no_save:
        logger.success(f"Saved 3D Biplot -> {png_path!r}")
    else:
        logger.info("Displayed Interactive 3D PCA Biplot in browser.")


if __name__ == "__main__":
    app()
