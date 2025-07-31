# kmflow

Modular data segmentation using PCA and K-Means in Python

## Overview

**kmflow** is a lightweight Python package and CLI toolset for exploring, segmenting, and visualizing tabular data via principal-component analysis (PCA) and K-Means clustering. It also ships with a Streamlit dashboard (`app/app.py`) for interactive analysis.

Key features:

- **CLI workflows** for preprocessing, clustering, PCA, evaluation & plotting  
- **Streamlit dashboard** for point-and-click analysis and download of results  
- **Modular utils** for wrangling, scaling, PCA summary, cluster prep, benchmarks  
- **Cookiecutter-inspired layout** that’s been customized for clarity and flexibility  

---

## Quickstart

1. **Clone the repo**

   ```bash
   $ git clone git@github.com:you/kmflow.git
   $ cd kmflow
   ```

2. **Create & activate a virtual environment**

   ```bash
   # Create venv
   $ python -m venv .venv

   # Activate
   $ source .venv/bin/activate    # macOS / Linux
   ```

3. **Install kmflow package** *(editable mode)*
   ```bash
   $ pip install -e .
   ```

4. **Install dependencies**

   ```bash
   $ pip install -r requirements.txt

   # If pip isn't installed, run this and then install dependencies
   $ pip install -U pip
   ```

5. **Run the dashboard if you want a more user-friendly experience** *(Optionally)*

   ```bash
   $ cd app
   $ streamlit run app.py
   ```
   This will launch and host the dashboard on your local machine. Alternatively, you can skip
   hosting it locally and use the hosted version by following this link:
   [K-Means Clustering Dashboard](https://kmeans-dash.streamlit.app/)

6. **Check out the *Description* below for a detailed breakdown and examples of how to use the CLI
   if you'd like to go beyond the dashboard experience.**

   
## Project Organization

```
├── app/                          
│   └── app.py                   # Streamlit dashboard entrypoint  
├── docs/                        # A default mkdocs project
│   └── mkdocs.md  
├── kmflow/                      # Python package  
│   ├── cli/                     # Command-line interface  
│   ├── utils/                   # Helper modules (wrangle, scale, PCA, plots, eval)  
│   └── config.py                # Global constants & defaults  
├── main/                        
│   └── entrypoint.py            # Console‐script entry point: builds the `kmflow` Typer app
|                                  by importing and registering all subcommands from kmflow/cli 
├── tests/                       
│   └── test_data.py             # Basic unit tests  
├── LICENSE                      
├── Makefile                     # `make data`, `make train`, etc.  
├── pyproject.toml               # Packaging + tool config (black, isort, etc.)  
├── README.md                    # You are here  
└── requirements.txt             # Pinned dependencies  

```
--------

## Description

### `__init__.py`

- Three different files
  - Top-level
  - util
  - CLI

| File                                  | Purpose                                                      |
|---------------------------------------|--------------------------------------------------------------|
| `kmflow/__init__.py`                  | Package initializer—defines `__version__`, exposes top-level API and makes `kmflow` importable. |
| `kmflow/cli/__init__.py`              | CLI entry-point setup—imports and registers all subcommands into a single Typer app.           |
| `kmflow/utils/__init__.py`            | Utility namespace—collects and exposes core helper functions (`wrangle`, `kmeans`, `pca`, etc.) for easy import. |

### `kmflow/cli/wrangle.py` & `kmflow/utils/wrangle_utils.py`

- **Purpose:** basic data-wrangling and IQR-based outlier handling as a prelude to scaling and clustering.  
- **CLI (`wrangle.py`):** provides flags to detect or remove interquartile-range outliers and invoke any custom cleaning steps you’ve added.  
- **Utils (`wrangle_utils.py`):** core functions for:
  - IQR outlier detection & optional removal  
  - simple cleaning helpers (e.g. drop columns, drop rows)  
- **Extensibility:** these modules form a scaffold—you’re expected to add dataset-specific wrangling functions (e.g. custom parsers, imputation routines) and register them in the CLI so every unique “dirty” case can be handled.  
- **When to use:** run this first on raw data to ensure all downstream steps (scaling, PCA, K-Means) receive clean, consistent inputs.

#### **Example usage in bash**

**Note:** When directing output to a custom destination (`-o -`) with multiple CSV-output flags (e.g. `--export-outliers` and `--remove-outliers`), kmflow concatenates all generated CSV outputs into a single file or stream. This behavior applies to any CLI tool in kmflow that supports more than one CSV-output flag.

- Using stdin -> default stdout (no `-o` flag)

```bash
$ cat data/raw/example.csv \
    | kmflow wrangle outlier - \
      --export-outliers \
      --remove-outliers
```

- Using stdin -> custom stdout (`-o` -)
  
```bash
$ cat data/raw/example.csv \
  | kmflow wrangle outlier - \
    --export-outliers \
    --remove-outliers \
    -o - > ../../data/DIR/output.csv
```

### `kmflow/cli/plots.py` & `kmflow/utils/plots_utils.py`

- **Purpose:** A suite of diagnostic and exploratory visualizations for clustering and PCA results, from 2D/3D biplots to inertia and silhouette curves.  
- **CLI (`plots.py`):** Entry-point script exposing commands (`biplot`, `3d-biplot`, `inertia`, `silhouette`, etc.). Reads input via file paths or stdin (`-`), accepts plot-specific flags (e.g. `--hue`, `--skip-scores`, etc.) and writes output via `-o` to files or default stdout.  
- **Utils (`plots_utils.py`):** Core plotting functions (using matplotlib, seaborn, and plotly under the hood) that handle data formatting, layout/styling, and figure saving. Includes shared helpers for colors and axis formatting.  

#### **Example usage in bash**

- Using stdin -> default stdout (no `-o` flag)

```bash
$ cat data/raw/example.csv \
    | kmflow plots histogram - COLUMN \
      --bins 50 \
      --no-save
```

- Using stdin -> custom stdout (`-o` -)
  
```bash
$ cat data/raw/example.csv \
  | kmflow plots histogram - COLUMN \
    --bins 25 \
    -o - > ../../data/DIR/histogram.png
```

### `kmflow/cli/process.py` & `kmflow/utils/process_utils.py`

- **Purpose:** Apply configurable scaling and transformations (normalization, standardization, min–max, log1p, Yeo–Johnson) to prepare data for PCA and K-Means.  
- **CLI (`process.py`):** Entry-point for data processing—reads from file or stdin (`-`), accepts flags for one or more transformations (e.g. `--standardize`, `--minmax`, `--log1p`, `--yeo-johnson`), and writes the transformed DataFrame to file or stdout (`-o`).  
- **Utils (`process_utils.py`):** Implements core scaling functions, handles column selection and transformer fitting, and integrates with I/O helpers for seamless DataFrame processing.  
- **When to use:** Run immediately after wrangling raw data to ensure features are on comparable scales before PCA or K-Means clustering if your data is not already on a similar scale.

#### **Example usage in bash**

- Using stdin -> default stdout (no `-o` flag)
  ```bash
  $ cat data/raw/example.csv \
    | kmflow process - \
      --standardize
  ```

- Using stdin -> custom stdout (`-o` -)

  ```bash
  $ cat data/raw/example.csv \
  | kmflow process - \
    --standardize \
    -o - > ../../data/DIR/std.csv
  ```

  ### `kmflow/cli/pca.py` & `kmflow/utils/pca_utils.py`

- **Purpose:** Perform principal component analysis to reduce dimensionality and extract component scores, loadings, proportion of explained variance, and cumulative variance.  
- **CLI (`pca.py`):** Entry-point that reads a processed DataFrame (file path or stdin `-`), accepts flags like `--n-components <int>`, and writes a PCA summary (CSV) to file or stdout (`-o`).  
- **Utils (`pca_utils.py`):** Implements PCA decomposition (using scikit-learn), computes scores, loadings, proportion of variance, and cumulative variance explained.  
- **When to use:** After preprocessing (wrangling and scaling) — and optionally clustering — to explore principal components and feed them into downstream analyses.

#### **Example usage in bash**

- Using stdin -> default stdout (no `-o` flag)  

  ```bash
  $ cat data/processed/DIR/std.csv \
    | kmflow pca - \
      --seed 3429 \
      --numeric-cols "COLUMN, COLUMN2, COLUMN3" \
      --n-components 3 
  ```
  
- Using stdin -> custom stdout (`-o` -)
  
  ```bash
  $ cat data/processed/DIR/std.csv \
    | kmflow pca - \
      -o - > ../../data/processed/DIR/std_pca.csv
  ```

### `kmflow/cli/kmeans.py` & `kmflow/utils/kmeans_utils.py`

- **Purpose:** Fit K-Means clustering to your data, assigning cluster labels and optionally running batch fits over multiple `k` values.  
- **CLI (`kmeans.py`):** Entry-point that reads a DataFrame (file path or stdin `-`), accepts clustering flags (`--n-clusters`, `--init-method`, `--seed`, etc.), and writes the labeled DataFrame to file or stdout (`-o`).  
- **Utils (`kmeans_utils.py`):** Core functions that wrap scikit-learn’s KMeans, handle single- or multi-`k` fits, manage random-state reproducibility, and append cluster labels to the DataFrame.  
- **When to use:** Run after data has been wrangled and scaled to segment into groups before evaluation or plotting.

#### **Example usage in bash**

- Using stdin -> default stdout (no `-o` flag)  
  ```bash
  $ cat data/processed/DIR/std.csv \
    | kmflow kmeans fit-km - 8 \
      --seed 123
      --init-method k-means++ 
  ```

  - Using stdin -> custom stdout (`-o` -)
  ```bash
  $ cat data/processed/DIR/std.csv \
    | kmflow kmeans fit-km - 8 \
      --seed 9250 \
      --algorithm elkan \
      --init random \
      --n-init 100 \
      -o - > ../../data/processed/DIR/clustered_8.csv
  ```

### `kmflow/cli/evaluation.py` & `kmflow/utils/evaluation_utils.py`

- **Purpose:** Compute clustering evaluation metrics (inertia, silhouette score, Calinski–Harabasz index, Davies–Bouldin score).  
- **CLI (`evaluation.py`):** Reads a processed DataFrame via file path or stdin (`-`). Then writes a CSV to stdout or the specified file.  
- **Utils (`evaluation_utils.py`):** Core functions that calculate each metric, assemble them into a tidy DataFrame, and hand off to I/O helpers.  
- **When to use:** Immediately after all clustering is finished to generate a quantitative report (see "Benchmarking Multiple Metrics" section below).

#### **Example usage in bash**

- Using stdin -> default stdout (no `-o` flag)
  ```bash
  $ cat data/processed/DIR/std.csv \
    | kmflow evaluate silhouette - \
      --seed 2985 \ 
      --algorithm elkan \
      --init random \
      --n-init 50
  ```

- Using stdin -> custom stdout (`-o` -)

  ```bash
  $ cat data/processed/DIR/std.csv \
    | kmflow evaluate silhouette - \
      --seed 2985 \
      -o - > ../../data/processed/DIR/std_silhouette.csv
  ```

  #### Benchmarking Multiple Metrics
  
  **kmflow** also provides a `benchmark` command to Calinski–Harabasz and Davies–Bouldin CSVs, merges them
  into one table, and writes to stdout or a file. It uses this *regex* to discover result folders:

  ```python
  r"^algo_([^_]+)_init_(.+)$"
  ```

  #### Directory & file naming conventions
  ```
  data/processed_root/
  └── <variant>/                      
      └── kmeans_<pipeline>/          
          └── algo_<algorithm>_init_<init>/  
              ├── <input_stem>_calinski.csv  
              └── <input_stem>_davies.csv
  ```

 - `<variant> = your processing variant (e.g. `std`, etc.)
 - `algo_<algorithm>_init_<init>` *must* match the regex above
 - **Files** *must* end in `_calinski.csv` or `_davies.csv`

  #### Merge via the CLI

  ```bash
  $ kmflow evaluate benchmark <input_dir> [flags]
  ```

- `<input_dir>` = subfolder under `data/` (e.g. `processed_std`)
- `--output-file, -o` = destination file or use `-` for stdout (default)
- `--decimals, d` = round metric values (default 3 decimal places)

#### **Example usage in bash**

- Write to stdout
  
  ```bash
  $ kmflow evaluate benchmark processed \
    --decimals 4
  ```

- Write to a file

  ```bash
  $ kmflow evaluate benchmark processed \
  -o - > ../../data/external/benchmark.csv
  ```

  ### `kmflow/cli/cluster_prep.py` & `kmflow/utils/cluster_prep_utils.py`

- **Purpose:** Merge raw and clustered DataFrames to produce per-cluster summary statistics and label counts.  
- **CLI (`cluster_prep.py`):**  
  - `cluster-profiles`: reads a raw CSV and a clustered CSV (file paths or `-` for stdin), accepts `--cluster-col` (and optional `--key-col` to join on a key column), and writes a per-cluster profile table.  
  - `map-clusters`: reads a clustered CSV, prompts interactively to map integer IDs to human labels, counts each label, and writes the counts table.  
- **Utils (`cluster_prep_utils.py`):**  
  - `merge_cluster_labels()` aligns raw & clustered rows by index (or key column).  
  - `get_cluster_profiles()` computes summary statistics (mean, median, min, max) per cluster.  
  - `clusters_to_labels()` and `count_labels()` map IDs to labels and tally counts.  
- **When to use:** After K-Means clustering, to turn raw + cluster outputs into interpretable summaries and human-readable labels.

#### Why?

If your data is not on a similar scale, K-Means and PCA won't perform well, but scaled data loses its original units—making it hard to interpret clusters in context. 
The **cluster_prep** tools let you reattach cluster assignments to the *unscaled* raw data so you can:

- Compute summary statistics in the original measurement units  
- Label clusters meaningfully based on real-world values  
- Count and compare cluster sizes accurately  
- Preserve the benefits of scaling for analysis while keeping outputs interpretable

#### **Example usage in bash**

**Note**: While stdin -> stdout piping is supported, I *highly* recommend using explicit file arguments as shown below.

- Write to stdout 

  ```bash
  $ kmflow cluster-prep cluster-profiles \
    ../../data/raw/DIR/example.csv \
    ../../data/processed/DIR/variant/algo_<algorithm>_init_<init>/std_clustered_5.csv \
    cluster_5
  ```

- Write to a file

  ```bash
  $ kmflow cluster-prep cluster-profiles \
    ../../data/raw/DIR/example.csv \
    ../../data/processed/DIR/variant/algo_<algorithm>_init_<init>/std_clustered_5.csv \
    cluster_5 \
    -o - > ../../data/external/cluster-profile.csv
  ```
  


      

  
     


  





























