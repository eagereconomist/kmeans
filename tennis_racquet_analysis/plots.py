import typer
import matplotlib.pyplot as plt
from loguru import logger
from tqdm import tqdm
from tennis_racquet_analysis.plots_utils import histogram

app = typer.Typer()


@app.command("hist")
def hist(
    dir_label: str,
    file_label: str,
    output_path: str,
    x_axis: str,
    num_bins: int,
):
    logger.info(f"Histogram: '{x_axis}' from {file_label} in directory '{dir_label}'.")
    progress_bar = tqdm(total=3, desc="Histogram Steps", ncols=80)
    progress_bar.set_description("Reading csv...")
    df = histogram(dir_label, file_label, output_path, x_axis, num_bins)
    progress_bar.update(1)
    progress_bar.set_description("Plotting & Saving...")
    progress_bar.update(2)
    progress_bar.close()
    logger.success(f"Saved histogram to {output_path}!")
    plt.show()
    return df


if __name__ == "__main__":
    app()
