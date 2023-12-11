import tensorflow as tf
import numpy as np
import keras

from Functions.Dataset_Treatment.dataset_treatment import normalize_dataset, dividedataset_oneyearperiod
from Functions.Modeling.multivariable_modeling import create_windows_multivariate_np
from Functions.NeuralNetworks.neuralnetworks_functions import (create_deeplstmmodel, create_deepgrumodel,
                                                               create_deepmodel, create_deepconvmodel,
                                                               create_simplernnmodel)


def generate_predictions_multivariable(df, test_data, window, horizon, best_model, target):
    # Parameters
    target_col_idx = df.columns.get_loc(target)

    # Divide in blocks of one year, result = list of vectors
    test_datadivide, itermax = dividedataset_oneyearperiod(test_data)

    # Normalize each block and save his mu and sigma
    mu_test = []
    sigma_test = []

    test_datanorm = test_datadivide.copy()

    for i in range(0, itermax):
        # Normalize test_data
        test_datanorm[i], mu_block, sigma_block = normalize_dataset(test_datadivide[i])

        # Save the metrics of normalization
        mu_test.append(mu_block)
        sigma_test.append(sigma_block)

    # Save the mu_target and sigma_target
    mu_target = [mu_test[x][target] for x in range(0, itermax)]
    sigma_target = [sigma_test[x][target] for x in range(0, itermax)]

    # Loop for every block
    # --------------------
    for block in range(0, itermax):
        # Create windows of the block
        X_testaux, y_testaux = create_windows_multivariate_np(test_datanorm[block], window, horizon,
                                                              target_col_idx=target_col_idx, shuffle=False)

        # Predict with X_testaux -> y_predaux
        y_predaux = best_model.predict(X_testaux, verbose=0)

        # Unormalize y_predaux con mu_target and sigma_target
        y_predaux = y_predaux * sigma_target[block] + mu_target[block]

        # Unormalize y_testaux con mu_target and sigma_target
        y_testaux = y_testaux * sigma_target[block] + mu_target[block]

        # Save in an array ypred and y_test
        if block == 0:
            y_pred = y_predaux
            y_test = y_testaux
        else:
            y_pred = np.concatenate((y_pred, y_predaux))
            y_test = np.concatenate((y_test, y_testaux))

    # Return y_pred e y_test
    return y_pred, y_test


def fit_and_generate_predictions_multivariable(df, train_data, test_data, window, horizon, network, target):
    # Parameters settings
    num_features = len(df.columns)
    target_col_idx = df.columns.get_loc(target)

    # TREAT TRAIN DATASET
    # -------------------
    # Normalize and save mu, sigma of the columns
    train_datanorm = train_data.copy()
    train_datanorm, mu_train, sigma_train = normalize_dataset(train_data)

    # Create windows
    X_train, y_train = create_windows_multivariate_np(train_datanorm, window, horizon, target_col_idx=target_col_idx,
                                                      shuffle=True)

    # TREAT TEST DATASET
    # -------------------
    # Divide in blocks of one year, result = list of vectors
    test_datadivide, itermax = dividedataset_oneyearperiod(test_data)

    # Normalize each block and save his mu and sigma
    mu_test = []
    sigma_test = []

    test_datanorm = test_datadivide.copy()

    for i in range(0, itermax):
        # Normalize test_data
        test_datanorm[i], mu_block, sigma_block = normalize_dataset(test_datadivide[i])

        # Save the metrics of normalization
        mu_test.append(mu_block)
        sigma_test.append(sigma_block)

    # Save the mu_target and sigma_target
    mu_target = [mu_test[x][target] for x in range(0, itermax)]
    sigma_target = [sigma_test[x][target] for x in range(0, itermax)]

    # Define the model
    # ----------------
    if network == 'LSTM':
        model6 = create_deeplstmmodel(neuron_l1=128, neuron_l2=128, neuron_l3=0,
                                      window_size=window, n_features=num_features)

    elif network == 'GRU':
        model6 = create_deepgrumodel(neuron_l1=128, neuron_l2=128, neuron_l3=0, neuron_l4=0,
                                     window_size=window, n_features=num_features)

    elif network == 'SimpleRNN':
        model6 = create_simplernnmodel(neuron_l1=128, neuron_l2=128, neuron_l3=0, neuron_l4=0,
                                       window_size=window, n_features=num_features)

    elif network == 'Convolutional':
        model6 = create_deepconvmodel(filters_l1=256, filters_l2=128, filters_l3=128,
                                      window_size=window, n_features=num_features)

    elif network == 'FeedForward':
        model6 = create_deepmodel(neuron_l1=256, neuron_l2=256, neuron_l3=256, neuron_l4=0,
                                  window_size=window, n_features=num_features)

    else:
        print('Error: The network case it is not defined')
        return None

    # Define the compilation settings
    # -------------------------------
    model6.compile(optimizer='adam', loss=tf.keras.losses.MSE, metrics=['mae'])

    # Fit the model with X_train and y_train
    # --------------------------------------
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)
    print('Starting the training of the multivariable network')
    print("-" * 60)
    history = model6.fit(X_train, y_train, epochs=200, validation_split=0.2, shuffle=True,
                         batch_size=128, callbacks=[es_callback], verbose=1)

    # Generate the predictions
    # ------------------------
    # Loop for every block
    # --------------------
    for block in range(0, itermax):
        # Create windows of the block
        X_testaux, y_testaux = create_windows_multivariate_np(test_datanorm[block], window, horizon,
                                                              target_col_idx=target_col_idx, shuffle=False)

        # Predict with X_testaux -> y_predaux
        y_predaux = model6.predict(X_testaux, verbose=0)

        # Unormalize y_predaux con mu_target and sigma_target
        y_predaux = y_predaux * sigma_target[block] + mu_target[block]

        # Unormalize y_testaux con mu_target and sigma_target
        y_testaux = y_testaux * sigma_target[block] + mu_target[block]

        # Save in an array ypred and y_test
        if block == 0:
            y_pred6 = y_predaux
            y_test = y_testaux
        else:
            y_pred6 = np.concatenate((y_pred6, y_predaux))
            y_test = np.concatenate((y_test, y_testaux))

    # Return the predictions
    # ----------------------------
    return y_pred6, y_test, model6
