import typer
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from tennis_racquet_analysis.config import DATA_DIR, FIGURES_DIR
from tennis_racquet_analysis.preprocessing_utils import load_data
from tennis_racquet_analysis.plots_utils import (
    histogram,
    scatter_plot,
    box_plot,
    violin_plot,
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


if __name__ == "__main__":
    app()
