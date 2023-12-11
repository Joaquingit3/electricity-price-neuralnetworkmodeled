import pandas as pd


# Function to divide a vector in blocks of one year each
def divide_oneyearperiod(X):
    X_divide = []

    # Find the number of year
    n_years = int(len(X) / 365)
    resto_years = len(X) % 365

    if resto_years == 0:
        for i in range(1, n_years + 1):
            X_divide.append(X[(i - 1)*365: 364 + (i - 1)*365 + 1])

        return X_divide, n_years

    else:
        for i in range(1, n_years + 1):
            X_divide.append(X[(i - 1)*365: 364 + (i - 1)*365 + 1])

            # Add the iterations of the last year
            X_divide.append(X[(n_years*365):])

        return X_divide, (n_years + 1)


def dividedataset_oneyearperiod(X):
    X_divide = []

    # Find the number of year
    n_years = int(X.shape[0] / 365)
    resto_years = X.shape[0] % 365

    if resto_years == 0:
        for i in range(1, n_years + 1):
            extrem_i = (i - 1)*365
            extrem_j = 364 + (i - 1)*365 + 1
            X_divide.append(X.iloc[extrem_i:extrem_j])

        return X_divide, n_years

    else:
        for i in range(1, n_years + 1):
            extrem_i = (i - 1)*365
            extrem_j = 364 + (i - 1)*365 + 1
            X_divide.append(X.iloc[extrem_i:extrem_j])

        # Add the iterations of the last year
        extrem_i = (n_years*365)
        X_divide.append(X.iloc[extrem_i:])

        return X_divide, (n_years + 1)


def normalize_vector(X):
    # Calculate mu and sigma
    mu = X.mean()
    sigma = X.std()

    # Normalize the vector
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma


def normalize_dataset(X):
    # Find the number of columns of the dataset
    features = X.columns

    # Auxiliar dictionaries
    mu_dict = {}
    sigma_dict = {}

    # Loop of the features to find mu and sigma
    for c in features:
        mu = X[c].mean()
        sigma = X[c].std()
        mu_dict[c] = mu
        sigma_dict[c] = sigma

    # Normalize the features with their mu and sigma
    X_norm = (X[features] - mu_dict) / sigma_dict

    # Return the dataset normalize
    return X_norm, mu_dict, sigma_dict


def split_train_test(df, split_fraction):
    # Create the Train index
    train_split = int(split_fraction * int(len(df)))

    # Create the train and test dfs
    train_data = df.iloc[0:train_split]
    test_data = df.iloc[train_split:]

    # Show the dimensions of train and test data
    print("Train-Test Split Dimensions")
    print("-"*32)
    print("dataset_train shape = ", train_data.shape)
    print("dataset_test shape = ", test_data.shape)
    print("\n")

    # Return the train and test
    return train_data, test_data


def import_dataset(data_location):
    # Read the file with pandas
    df = pd.read_csv(data_location, parse_dates=['time'])

    # Set time as index
    df = df.set_index('time')

    # Return df
    return df
