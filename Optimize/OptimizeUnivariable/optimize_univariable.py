import keras
import tensorflow as tf
import numpy as np

from Functions.Dataset_Treatment.dataset_treatment import normalize_vector, divide_oneyearperiod
from Functions.Modeling.univariable_modeling import create_windows_np
from Functions.NeuralNetworks.neuralnetworks_functions import (create_deeplstmmodel, create_deepgrumodel,
                                                               create_simplernnmodel, create_deepconvmodel)


def generate_predictions_univariable(window, horizon, test_data, best_model):
    # Divide in blocks of one year, result = list of vectors
    test_datadivide, itermax = divide_oneyearperiod(test_data)

    # Normalize each block and save his mu and sigma
    mu_test = []
    sigma_test = []

    test_datanorm = test_datadivide.copy()

    for i in range(0, itermax):
        # Normalize test_data
        test_datanorm[i], mu_block, sigma_block = normalize_vector(test_datadivide[i])

        # Save the metrics of normalization
        mu_test.append(mu_block)
        sigma_test.append(sigma_block)

    # Save the mu_target and sigma_target
    mu_target = [mu_test[x] for x in range(0, itermax)]
    sigma_target = [sigma_test[x] for x in range(0, itermax)]

    # Loop for every block
    # --------------------
    for block in range(0, itermax):
        # Create windows of the block
        X_testaux, y_testaux = create_windows_np(test_datanorm[block], window, horizon, shuffle=False)

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


def generate_predictions_lstm_univariable(train_data, test_data, window, horizon):
    # Parameters settings
    num_features = 1

    # TREAT TRAIN DATASET
    # -------------------
    # Normalize and save mu, sigma of the columns
    train_datanorm = train_data.copy()
    train_datanorm, mu_train, sigma_train = normalize_vector(train_data)

    # Create windows
    X_train, y_train = create_windows_np(train_datanorm, window, horizon, shuffle=True)

    # Define the model
    # ----------------
    model2 = create_deeplstmmodel(neuron_l1=128, neuron_l2=128, neuron_l3=0,
                                  window_size=window, n_features=num_features)

    # Define the compilation settings
    # -------------------------------
    model2.compile(optimizer='adam', loss=tf.keras.losses.MSE, metrics=['mae'])

    # Fit the model with X_train and y_train
    # --------------------------------------
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)
    print('Starting training of the LSTM network')
    print("-" * 42)
    history = model2.fit(X_train, y_train, epochs=200, validation_split=0.2, shuffle=True,
                         batch_size=128, callbacks=[es_callback], verbose=1)

    # Generate the predictions of the model
    # -------------------------------------
    y_pred2, y_test = generate_predictions_univariable(window, horizon, test_data, model2)

    # Return the predictions, the test and the model
    return y_pred2, y_test, model2


def generate_predictions_gru_univariable(train_data, test_data, window, horizon):
    # Parameters settings
    num_features = 1

    # TREAT TRAIN DATASET
    # -------------------
    # Normalize and save mu, sigma of the columns
    train_datanorm = train_data.copy()
    train_datanorm, mu_train, sigma_train = normalize_vector(train_data)

    # Create windows
    X_train, y_train = create_windows_np(train_datanorm, window, horizon, shuffle=True)

    # Define the model
    # ----------------
    model3 = create_deepgrumodel(neuron_l1=128, neuron_l2=128, neuron_l3=0, neuron_l4=0,
                                 window_size=window, n_features=num_features)

    # Define the compilation settings
    # -------------------------------
    model3.compile(optimizer='adam', loss=tf.keras.losses.MSE, metrics=['mae'])

    # Fit the model with X_train and y_train
    # --------------------------------------
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)
    print('Starting the training of the GRU network')
    print("-" * 42)
    history = model3.fit(X_train, y_train, epochs=200, validation_split=0.2, shuffle=True,
                         batch_size=128, callbacks=[es_callback], verbose=1)

    # Generate the predictions
    # ------------------------
    y_pred3, y_test = generate_predictions_univariable(window, horizon, test_data, model3)

    # Return the predictions
    # ------------------------
    return y_pred3, y_test, model3


def generate_predictions_simplernn_univariable(train_data, test_data, window, horizon):
    # Parameters settings
    num_features = 1

    # TREAT TRAIN DATASET
    # -------------------
    # Normalize and save mu, sigma of the columns
    train_datanorm = train_data.copy()
    train_datanorm, mu_train, sigma_train = normalize_vector(train_data)

    # Create windows
    X_train, y_train = create_windows_np(train_datanorm, window, horizon, shuffle=True)

    # Define the model
    # ----------------
    model4 = create_simplernnmodel(neuron_l1=128, neuron_l2=128, neuron_l3=0, neuron_l4=0,
                                   window_size=window, n_features=num_features)

    # Define the compilation settings
    # -------------------------------
    model4.compile(optimizer='adam', loss=tf.keras.losses.MSE, metrics=['mae'])

    # Fit the model with X_train and y_train
    # --------------------------------------
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)
    print('Starting the training of the SimpleRNN network')
    print("-" * 42)
    history = model4.fit(X_train, y_train, epochs=200, validation_split=0.2, shuffle=True,
                         batch_size=128, callbacks=[es_callback], verbose=1)

    # Generate the predictions
    # ------------------------
    y_pred4, y_test = generate_predictions_univariable(window, horizon, test_data, model4)

    # Return the predictions
    # ------------------------
    return y_pred4, y_test, model4


def generate_predictions_conv_univariable(train_data, test_data, window, horizon):
    # Parameters settings
    num_features = 1

    # TREAT TRAIN DATASET
    # -------------------
    # Normalize and save mu, sigma of the columns
    train_datanorm = train_data.copy()
    train_datanorm, mu_train, sigma_train = normalize_vector(train_data)

    # Create windows
    X_train, y_train = create_windows_np(train_datanorm, window, horizon, shuffle=True)

    # Define the model
    # ----------------
    model5 = create_deepconvmodel(filters_l1=256, filters_l2=128, filters_l3=128,
                                  window_size=window, n_features=num_features)

    # Define the compilation settings
    # -------------------------------
    model5.compile(optimizer='adam', loss=tf.keras.losses.MSE, metrics=['mae'])

    # Fit the model with X_train and y_train
    # --------------------------------------
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)
    print('Starting the training of the Convolutional network')
    print("-" * 42)
    history = model5.fit(X_train, y_train, epochs=200, validation_split=0.2, shuffle=True,
                         batch_size=128, callbacks=[es_callback], verbose=1)

    # Generate the predictions
    # ------------------------
    y_pred5, y_test = generate_predictions_univariable(window, horizon, test_data, model5)

    # Return the predictions
    # ------------------------
    return y_pred5, y_test, model5
