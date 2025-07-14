from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from kmflow.cli_utils import read_df
from kmflow.process_utils import (
    write_csv,
    apply_normalizer,
    apply_standardization,
    apply_minmax,
    log1p_transform,
    yeo_johnson,
)
from kmflow.config import PROCESSED_DATA_DIR

app = typer.Typer(help="Apply a suite of scalers to a preprocessed CSV.")


@app.command("process")
def process(
    input_file: Path = typer.Argument(
        ...,
        help="Feature-engineered CSV file to read (use '-' to read from stdin).",
    ),
    prefix: str = typer.Option(
        None,
        "--prefix",
        "-p",
        help="Base name for the output files (defaults to the input filename stem).",
    ),
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR,
        "--output-dir",
        "-o",
        dir_okay=True,
        file_okay=False,
        help="Directory where processed CSVs will be written.",
    ),
):
    """
    Read a feature-engineered CSV, apply multiple scalers, and write
    one output CSV per scaler:
      - <prefix>_normalized.csv
      - <prefix>_standardized.csv
      - <prefix>_minmax.csv
      - <prefix>_log_scale.csv
      - <prefix>_yeo_johnson.csv
    """
    # ─── 1) read input ───────────────────────────────────────────
    df = read_df(input_file)

    # ─── 2) determine prefix ─────────────────────────────────────
    if prefix is None:
        prefix = input_file.stem if input_file != Path("-") else "stdin"

    # ─── 3) apply each scaler ────────────────────────────────────
    steps = [
        ("normalized", apply_normalizer, "normalized"),
        ("standardized", apply_standardization, "standardized"),
        ("minmax", apply_minmax, "minmax"),
        ("log1p_transformed", log1p_transform, "log_scale"),
        ("yeo_johnson", yeo_johnson, "yeo_johnson"),
    ]
    processed_paths: dict[str, Path] = {}

    for name, func, suffix in tqdm(steps, desc="Scaling Steps", ncols=100):
        logger.info(f"Applying '{name}' scaler…")
        df_scaled = func(df.copy())
        out_path = write_csv(df_scaled, prefix=prefix, suffix=suffix, output_dir=output_dir)
        processed_paths[name] = out_path
        logger.success(f"Wrote {name} → {out_path!r}")

    typer.echo(f"All done! Files written to {output_dir!r}")
    return processed_paths


if __name__ == "__main__":
    app()
