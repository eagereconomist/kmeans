# Install necessary packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

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

print(df)

# Create deep copy of the data frame
df_preprocessed = df.copy()

# Index by 'Racquet'
index_racquet = df_preprocessed.set_index("Racquet")
print(index_racquet.head(10))

# Drop 'Racquet' column
df_preprocessed = df_preprocessed.drop(columns=["Racquet"])

# rename 'static.weight' column
df_preprocessed = df_preprocessed.rename(columns={"static.weight": "staticweight"})

# Add non-linear version of 'headsize'
df_preprocessed["headsize_sq"] = df_preprocessed["headsize"] ** 2

# Add non-linear version of 'swingweight'
df_preprocessed["swingweight_sq"] = df_preprocessed["swingweight"] ** 2

# Check the shape of the data frames
print(df_preprocessed.shape, df_normalized.shape)
