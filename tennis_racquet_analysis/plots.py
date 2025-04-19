import typer
import matplotlib.pyplot as plt
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
    dir_label: str = typer.Option(
        "processed", "--dir-label", "-d", help="Which sub-folder under data/ to read from."
    ),
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
    logger.info(f"Histogram of '{x_axis}' from '{dir_label}/{input_file}'")
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
    dir_label: str = typer.Option(
        "processed",
        "--dir-label",
        "-d",
        help="Data sub-folder.",
    ),
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
    logger.info(f"Scatterplot {x_axis} vs. {y_axis} from '{dir_label}/{input_file}'")
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
    dir_label: str = typer.Option("processed", "--dir-label", "-d", help="Data sub-folder."),
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
    logger.info(f"Box plot {x_axis} vs. {y_axis} from '{dir_label}/{input_file}'")
    box_plot(
        input_file=input_file,
        dir_label=dir_label,
        x_axis=x_axis,
        y_axis=y_axis,
        output_dir=output_dir,
    )
    logger.success(f"Box plot saved to {output_dir}!")


if __name__ == "__main__":
    app()
