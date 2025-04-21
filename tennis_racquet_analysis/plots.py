import typer
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from tennis_racquet_analysis.config import FIGURES_DIR
from tennis_racquet_analysis.plots_utils import (
    histogram,
    scatter_plot,
    box_plot,
)

app = typer.Typer()


@app.command("hist")
def hist(
    input_file: str = typer.Argument("csv filename."),
    dir_label: str = typer.Argument("Pick the parent data folder's sub-folder."),
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
):
    logger.info(
        f"Generating Histogram with {num_bins} bins for '{x_axis}' from '{dir_label}/{input_file}'"
    )
    histogram(
        input_file=input_file,
        dir_label=dir_label,
        x_axis=x_axis,
        num_bins=num_bins,
        output_dir=output_dir,
    )
    logger.success(f"Histogram saved to {output_dir}")


@app.command("scatter")
def scatter(
    input_file: str = typer.Argument(..., help="csv filename."),
    dir_label: str = typer.Argument("Pick the parent data folder's sub-folder."),
    x_axis: str = typer.Argument(..., help="X-axis column."),
    y_axis: str = typer.Argument(..., help="Y-axis column."),
    output_dir: Path = typer.Option(
        FIGURES_DIR,
        "--output-dir",
        "-o",
        dir_okay=True,
        file_okay=False,
    ),
):
    logger.info(f"Generating Scatterplot of {x_axis} vs. {y_axis} from '{dir_label}/{input_file}'")
    scatter_plot(
        input_file=input_file,
        dir_label=dir_label,
        x_axis=x_axis,
        y_axis=y_axis,
        output_dir=output_dir,
    )
    logger.success(f"Scatter plot saved to {output_dir}!")


@app.command("boxplot")
def boxplt(
    input_file: str = typer.Argument(..., help="csv filename."),
    dir_label: str = typer.Argument("Pick the parent data folder's sub-folder."),
    y_axis: str = typer.Argument(..., help="Y-axis column."),
    brand: str = typer.Option(
        None,
        "--brand",
        "-b",
        help="By default, 'brand' is None, but there is the option to pick a brand.",
    ),
    output_dir: Path = typer.Option(
        FIGURES_DIR,
        "--output-dir",
        "-o",
        dir_okay=True,
        file_okay=False,
    ),
):
    logger.info(
        f"Generating Box plot of {y_axis.capitalize()}' "
        f"{'for ' + brand if brand else 'by Brand'} "
        f"from ' {dir_label}/{input_file}'"
    )
    box_plot(
        input_file=input_file,
        dir_label=dir_label,
        y_axis=y_axis,
        brand=brand,
        output_dir=output_dir,
    )
    logger.success(f"Box plot saved to {output_dir}!")


if __name__ == "__main__":
    app()
