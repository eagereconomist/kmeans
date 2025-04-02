# Install necessary packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

# Build the relative path to your CSV file
data_path = os.path.join('data', 'raw', 'tennis-racquets.csv')
print("Looking for file at:", data_path)

# Check if the file exists before trying to load it
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    print("Data loaded successfully!")
    print(df.head())
else:
    print("File not found. Please check your path:", data_path)
