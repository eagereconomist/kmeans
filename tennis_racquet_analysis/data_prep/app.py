import module
import numpy as np
from sklearn import preprocessing


def main():
    """Load the raw data and run through preprocess pipeline"""
    preprocessed_data = (
        df.pipe(drop_column, "Racquet")
        .pipe(rename_column, "static.weight")
        .pipe(squared, "headsize")
        .pipe(squared, "swingweight")
        .pipe(write_csv, "interim", "preprocessed")
    )
    print("Data preprocessing complete")


if __name__ == "__main__":
    main()
