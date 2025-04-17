import typer
import matplotlib.pyplot as plt
from loguru import logger
from tqdm import tqdm
from tennis_racquet_analysis.plots_utils import histogram, scatter_plot

app = typer.Typer()


@app.command("hist")
def hist(
    dir_label: str,
    file_label: str,
    output_path: str,
    x_axis: str,
    num_bins: int,
):
    logger.info(
        f"Histogram: Plotting '{x_axis}' from {file_label} DataFrame in '{dir_label}' data folder."
    )
    progress_bar = tqdm(total=3, desc="Histogram Steps", ncols=80)
    progress_bar.set_description("Reading csv...")
    df = histogram(dir_label, file_label, output_path, x_axis, num_bins)
    progress_bar.update(1)
    progress_bar.set_description("Plotting & Saving...")
    progress_bar.update(2)
    progress_bar.close()
    logger.success(f"Saved histogram to {output_path} folder.")
    plt.show()
    return df


@app.command("scatter")
def scatter(
    dir_label: str,
    file_label: str,
    output_path: str,
    x_axis: str,
    y_axis: str,
):
    logger.info(
        f"Scatter plot: Plotting {x_axis} vs. {y_axis} from {file_label} DataFrame in '{dir_label}' data folder."
    )
    progress_bar_2 = tqdm(total=3, desc="Scatter Plot Steps", ncols=80)
    progress_bar_2.set_description("Reading csv...")
    df = scatter_plot(dir_label, file_label, output_path, x_axis, y_axis)
    progress_bar_2.update(1)
    progress_bar_2.set_description("Plotting & Saving...")
    progress_bar_2.update(2)
    progress_bar_2.close()
    logger.success(f"Saved scatter plot to {output_path} folder.")
    plt.show()
    return df


if __name__ == "__main__":
    app()
