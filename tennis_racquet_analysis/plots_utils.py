import numpy as np
import pandas as pd
from tennis_racquet_analysis.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from tennis_racquet_analysis.preprocessing_utils import load_data

import matplotlib.pyplot as plt
from itertools import groupby


colors = {
    "length": "#2B5B84",
    "staticweight": "green",
    "balance": "purple",
    "swingweight": "red",
    "headsize": "blue",
    "beamwidth": "orange",
}

plt.hist(load_data["length"], bins=30, color="blue", edgecolor="black")
plt.xlabel("Length")
plt.ylabel("Frequency")
plt.title("Histogram of Tennis Racquet Length")
plt.show()
