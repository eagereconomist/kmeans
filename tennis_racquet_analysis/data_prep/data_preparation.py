# Install necessary packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Build the relative path to CSV file
data_path = os.path.join("data", "raw", "tennis_racquets.csv")
print("Looking for file at:", data_path)

# Check if the file exists before trying to load it
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    print("Data loaded successfully!")
    print(df.head(10))
else:
    print("File not found. Please check your path:", data_path)

# See the first lines of the file
with open(data_path) as lines:
    for _ in range(10):
        print(next(lines))


def drop_column(dataframe, column):
    return dataframe.drop(columns=[column])


def rename_column(dataframe, column):
    new_column = column.replace(".", "")
    return dataframe.rename(columns={column: new_column})


def squared(dataframe, column) -> int:
    dataframe[f"{column}_sq"] = dataframe[column] ** 2
    return dataframe


def write_csv(dataframe, subfolder, file_label):
    file_path = os.path.join("data", subfolder, f"tennis_racquet_{file_label}.csv")
    dataframe.to_csv(file_path, index=False)
    return dataframe


preprocessed_data = (
    df.pipe(drop_column, "Racquet")
    .pipe(rename_column, "static.weight")
    .pipe(squared, "headsize")
    .pipe(squared, "swingweight")
    .pipe(write_csv, "interim", "preprocessed")
)
