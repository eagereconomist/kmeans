import module


def main():
    # Load raw data
    df = module.load_data()

    # Data Preprocessing Pipeline
    preprocessed_data = (
        df.pipe(module.drop_column, "Racquet")
        .pipe(module.rename_column, "static.weight")
        .pipe(module.squared, "headsize")
        .pipe(module.squared, "swingweight")
        .pipe(module.write_csv, "interim", "preprocessed")
    )
    print("Data preprocessing complete!")

    # Data Processing Pipelines using various scalers
    normalized_data = preprocessed_data.pipe(module.apply_normalizer).pipe(
        module.write_csv, "processed", "normalized"
    )

    standardized_data = preprocessed_data.pipe(module.apply_standardization).pipe(
        module.write_csv, "processed", "standardized"
    )

    minmax_data = preprocessed_data.pipe(module.apply_minmax).pipe(
        module.write_csv, "processed", "minmax"
    )

    log_scale_data = preprocessed_data.pipe(module.log_transform).pipe(
        module.write_csv, "processed", "log_scale"
    )

    yeo_johnson_data = preprocessed_data.pipe(module.yeo_johnson).pipe(
        module.write_csv, "processed", "yeo_johnson"
    )

    print("Data processing complete!")

    # Return a dictionary of the processed DataFrames
    return {
        "normalized": normalized_data,
        "standardized": standardized_data,
        "minmax": minmax_data,
        "log_scale": log_scale_data,
        "yeo_johnson": yeo_johnson_data,
    }


if __name__ == "__main__":
    main()
