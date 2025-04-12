def squared(dataframe, column):
    dataframe[f"{column}_sq"] = dataframe[column] ** 2
    return dataframe
