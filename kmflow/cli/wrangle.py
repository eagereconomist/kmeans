import sys
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from kmflow.cli_utils import read_df, comma_split
from kmflow.wrangle_utils import (
    find_iqr_outliers,
    drop_column,
    drop_row,
    dotless_column,
)
from kmflow.process_utils import write_csv
from kmflow.config import INTERIM_DATA_DIR

app = typer.Typer(help="Data preprocessing commands.")


@app.command("outlier")
def iqr_outliers(
    input_file: Path = typer.Argument(
        ...,
        help="CSV file to read (use '-' for stdin).",
    ),
    export_outliers: bool = typer.Option(
        False,
        "--export-outliers",
        "-eo",
        help="Write detected outliers to CSV.",
    ),
    remove_outliers: bool = typer.Option(
        False,
        "--remove-outliers",
        "-ro",
        help="Write cleaned CSV (with outliers removed).",
    ),
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory in which to save output CSVs; use '-' for stdout.",
    ),
):
    """
    Identify IQR-based outliers and optionally export and/or remove them.
    """
    total_steps = 2 + int(export_outliers) + int(remove_outliers)
    with tqdm(total=total_steps, desc="IQR Outliers", colour="green") as pbar:
        df = read_df(input_file)
        pbar.update(1)

        out = find_iqr_outliers(df)
        pbar.update(1)

        if out.empty:
            logger.info("No IQR-based outliers detected.")
            return

        out_df = out.reset_index().rename(
            columns={"level_0": "row_index", "level_1": "column", 0: "outlier_value"}
        )

        if output_dir is None:
            output_dir = INTERIM_DATA_DIR

        stem = input_file.stem if input_file != Path("-") else "stdin"

        if output_dir == Path("-") and export_outliers and remove_outliers:
            logger.warning(
                "You're exporting both outliers and cleaned data to stdout in one stream. "
                "They'll be concatenated. If you want separate files,\n "
                "omit '-o - ' so you write two files into the default directory."
            )

        if not remove_outliers and export_outliers:
            count = out_df.shape[0]
            logger.info(
                f"{count} IQR outliers detected. "
                "Use --remove-outliers (Shorthand: -ro) to identify and remove them."
            )
        if export_outliers:
            if output_dir == Path("-"):
                out_df.to_csv(sys.stdout.buffer, index=False)
                logger.success("Detected IQR outlier data written to stdout.")
            else:
                path = write_csv(
                    out_df,
                    prefix=stem,
                    suffix="iqr_outliers",
                    output_dir=output_dir,
                )
                logger.success(f"Data written to {path!r}")
            pbar.update(1)
        if remove_outliers:
            rows = out_df["row_index"].unique().tolist()
            cleaned = drop_row(df, rows)
            if output_dir == Path("-"):
                cleaned.to_csv(sys.stdout.buffer, index=False)
                logger.success("IQR outliers removed and data written to stdout.")
            else:
                path = write_csv(
                    cleaned,
                    prefix=stem,
                    suffix="no_outliers",
                    output_dir=output_dir,
                )
                logger.success(f"Data written to {path!r}")
            pbar.update(1)


@app.command("preprocess")
def preprocess(
    input_file: Path = typer.Argument(
        ...,
        help="CSV to read (use '-' for stdin).",
    ),
    dropped_columns: str = typer.Option(
        "",
        "--dropped-column",
        "-dc",
        help="Columns to drop, comma-separated or repeatable.",
        callback=lambda x: comma_split(x) if isinstance(x, str) else x,
    ),
    dotless_columns: str = typer.Option(
        "",
        "--dotless-column",
        "-dot",
        help="Columns whose dots to remove, comma-separated or repeatable.",
        callback=lambda x: comma_split(x) if isinstance(x, str) else x,
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        "-p",
        help="Show first 5 rows of the preprocessed DataFrame.",
    ),
    output_file: Path = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Where to save the cleaned CSV; use '-' for stdout.",
    ),
):
    """
    Apply column/row drops and dotless renaming to a CSV. Does NOT handle IQR outliers.
    """
    df = read_df(input_file)
    stem = input_file.stem if input_file != Path("-") else "stdin"

    steps: list[tuple[str, callable, list]] = []
    for col in dropped_columns:
        steps.append(("drop_column", drop_column, [col]))
    for col in dotless_columns:
        steps.append(("dotless_column", dotless_column, [col]))

    for name, func, args in tqdm(steps, desc="Data Preprocessing Steps", colour="green"):
        logger.info(f"Applying {name}...")
        df = func(df, *args)

    if preview:
        typer.echo(df.head())

    if output_file is None:
        output_file = INTERIM_DATA_DIR / f"{stem}_preprocessed.csv"

    if output_file == Path("-"):
        df.to_csv(sys.stdout.buffer, index=False)
        logger.success("Preprocessed CSV written to stdout.")
    else:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        logger.success(f"Preprocessed CSV saved to {output_file!r}")

    return df
