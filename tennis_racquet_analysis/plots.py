import typer
import matplotlib.pyplot as plt
from loguru import logger
from tennis_racquet_analysis.plots_utils import histogram

app = typer.Typer()


@app.command()
def main(
    dir_label: str,
    file_label: str,
    output_path: str,
    x_axis: str,
    num_bins: int,
):
    logger.info(
        f"Creating histogram for column '{x_axis}' using file tennis_racquets_{file_label}.csv in the '{dir_label}' directory."
    )
    df = histogram(dir_label, file_label, output_path, x_axis, num_bins)
    logger.success(f"Histogram for '{x_axis}' created and figure saved to {output_path}!")
    plt.show()
    return df


if __name__ == "__main__":
    app()
