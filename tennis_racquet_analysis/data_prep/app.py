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


if __name__ == "__main__":
    main()
