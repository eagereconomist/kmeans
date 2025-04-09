import module


def main():
    """Load the raw data and run through preprocess pipeline"""
    df = module.load_data()

    module.preprocessed_data = (
        df.pipe(module.drop_column, "Racquet")
        .pipe(module.rename_column, "static.weight")
        .pipe(module.squared, "headsize")
        .pipe(module.squared, "swingweight")
        .pipe(module.write_csv, "interim", "preprocessed")
    )
    print("Data preprocessing complete")


normalized_data = module.preprocessed_data.pipe(module.apply_normalizer).pipe(
    module.write_csv, "processed", "normalized"
)

standardized_data = module.preprocessed_data.pipe(module.apply_standardization).pipe(
    module.write_csv, "processed", "standardized"
)

minmax_data = module.preprocessed_data.pipe(module.apply_minmax).pipe(
    module.write_csv, "processed", "minmax"
)

log_scale_data = module.preprocessed_data.pipe(module.log_transform).pipe(
    module.write_csv, "processed", "log_scale"
)

yeo_johnson_data = module.preprocessed_data.pipe(module.yeo_johnson).pipe(
    module.write_csv, "processed", "yeo_johnson"
)


if __name__ == "__main__":
    main()
