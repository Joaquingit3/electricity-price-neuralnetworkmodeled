import pandas as pd
import numpy as np


# Function to create a windows of multivariate df
def create_windows_multivariate_np(data, window_size, horizon, target_col_idx, shuffle=False):
    """
        Creates a dataset from the given time series data using NumPy.

        Parameters:
        - data (np.ndarray or pd.DataFrame): Time series data with multiple features.
        - window_size (int): The number of pastime steps to use as input features.
        - horizon (int): The number of future time steps to predict.
        - target_col_idx (int): The index of the target column in the input data.
        - shuffle (bool): Whether to shuffle the data or not.

        Returns:
        - tuple: A tuple containing the input-output pairs (X, y) as NumPy arrays.
    """
    if isinstance(data, pd.DataFrame):
        data = data.values

    X, y = [], []
    for i in range(len(data) - window_size - horizon + 1):
        X.append(data[i:i+window_size, :])
        y.append(data[i+window_size+horizon-1, target_col_idx])

    X, y = np.array(X), np.array(y)

    if shuffle:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]

    return X, y
