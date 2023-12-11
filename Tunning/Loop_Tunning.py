import numpy as np
import keras
import tensorflow as tf

from Functions.Dataset_Treatment.dataset_treatment import (normalize_vector, normalize_dataset,
                                                           divide_oneyearperiod, dividedataset_oneyearperiod)
from Functions.Modeling.univariable_modeling import create_windows_np
from Functions.Modeling.multivariable_modeling import create_windows_multivariate_np
from Functions.NeuralNetworks.neuralnetworks_functions import (create_deepmodel, create_deeplstmmodel,
                                                               create_deepgrumodel, create_simplernnmodel,
                                                               create_deepconvmodel)


def tune_with_loop_univariable(train_data, test_data, window, horizon, network):
    # Parameters settings
    num_features = 1

    neuron_l1 = [32, 64, 128, 256]
    neuron_l2 = [0, 32, 64, 128, 256]
    neuron_l3 = [0, 32, 64, 128, 256]

    # TREAT TRAIN DATASET
    # -------------------
    # Univariable case
    # Normalize and save mu, sigma of the columns
    train_datanorm, mu_train, sigma_train = normalize_vector(train_data)

    # Create windows
    X_train, y_train = create_windows_np(train_datanorm, window, horizon, shuffle=True)

    # TREAT TEST DATASET
    # -------------------
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

    # Auxiliar variables
    neuron1 = []
    neuron2 = []
    neuron3 = []
    test_mae = []
    test_rmse = []

    # Model and mae variables
    best_mae = float('inf')
    best_model = None

    # Early Stopping Callback
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

    # Loop to know the best model
    print('Starting the tuning of the {} network'.format(network))
    print('-'*50)
    for neuron_j in neuron_l2:
        for neuron_i in neuron_l1:
            if neuron_j == 0:
                # Define the model
                # ----------------
                if network == 'LSTM':
                    model2 = create_deeplstmmodel(neuron_l1=neuron_i, neuron_l2=neuron_j, neuron_l3=0,
                                                  window_size=window, n_features=num_features)

                elif network == 'GRU':
                    model2 = create_deepgrumodel(neuron_l1=neuron_i, neuron_l2=neuron_j, neuron_l3=0, neuron_l4=0,
                                                 window_size=window, n_features=num_features)

                elif network == 'SimpleRNN':
                    model2 = create_simplernnmodel(neuron_l1=neuron_i, neuron_l2=neuron_j, neuron_l3=0, neuron_l4=0,
                                                   window_size=window, n_features=num_features)

                elif network == 'Convolutional':
                    model2 = create_deepconvmodel(filters_l1=neuron_i, filters_l2=neuron_j, filters_l3=0,
                                                  window_size=window, n_features=num_features)

                elif network == 'FeedForward':
                    model2 = create_deepmodel(neuron_l1=neuron_i, neuron_l2=neuron_j, neuron_l3=0, neuron_l4=0,
                                              window_size=window, n_features=num_features)

                else:
                    print('Error: The network is not defined')
                    return None

                # Define the compilation settings
                # -------------------------------
                model2.compile(optimizer='adam', loss=tf.keras.losses.MSE, metrics=['mae'])

                # Fit the model with X_train and y_train
                # --------------------------------------
                history = model2.fit(X_train, y_train, epochs=200, validation_split=0.2, shuffle=True,
                                     batch_size=128, callbacks=[es_callback], verbose=0)

                # Loop for every block
                # --------------------
                mae_num = 0
                mse_num = 0
                mae_den = 0
                for block in range(0, itermax):
                    # Create windows of the block
                    X_testaux, y_testaux = create_windows_np(test_datanorm[block], window, horizon, shuffle=False)

                    # Evaluate the model
                    results = model2.evaluate(X_testaux, y_testaux, verbose=0)

                    # Construct the MAE
                    if block == 0:
                        mae_num = len(y_testaux) * sigma_target[block] * results[1]
                        mse_num = len(y_testaux) * sigma_target[block] * sigma_target[block] * results[0]
                        mae_den = len(y_testaux)

                    else:
                        mae_num = mae_num + len(y_testaux) * sigma_target[block] * results[1]
                        mse_num = mse_num + len(y_testaux) * sigma_target[block] * sigma_target[block] * results[0]
                        mae_den = mae_den + len(y_testaux)

                # Save the results of the combination
                mae = mae_num / mae_den
                mse = mse_num / mae_den

                neuron1.append(neuron_i)
                neuron2.append(neuron_j)
                neuron3.append(0)
                test_mae.append(mae)
                test_rmse.append(np.sqrt(mse))

                # Show the results
                print('Neuron_l1 = {:3d}, Neuron_l2 = {:3d}, Neuron_l3 = {:3d} | '
                      'Test RMSE: {:.3f}, Test MAE = {:.3f}'.format(neuron_i, neuron_j, 0, np.sqrt(mse), mae))

                # Save the best model
                if best_mae > mae:
                    best_model = model2
                    best_mae = mae

            else:
                for neuron_k in neuron_l3:
                    # Define the model
                    # ----------------
                    if network == 'LSTM':
                        model2 = create_deeplstmmodel(neuron_l1=neuron_i, neuron_l2=neuron_j, neuron_l3=neuron_k,
                                                      window_size=window, n_features=num_features)

                    elif network == 'GRU':
                        model2 = create_deepgrumodel(neuron_l1=neuron_i, neuron_l2=neuron_j,
                                                     neuron_l3=neuron_k, neuron_l4=0,
                                                     window_size=window, n_features=num_features)

                    elif network == 'SimpleRNN':
                        model2 = create_simplernnmodel(neuron_l1=neuron_i, neuron_l2=neuron_j,
                                                       neuron_l3=neuron_k, neuron_l4=0,
                                                       window_size=window, n_features=num_features)

                    elif network == 'Convolutional':
                        model2 = create_deepconvmodel(filters_l1=neuron_i, filters_l2=neuron_j, filters_l3=neuron_k,
                                                      window_size=window, n_features=num_features)

                    elif network == 'FeedForward':
                        model2 = create_deepmodel(neuron_l1=neuron_i, neuron_l2=neuron_j,
                                                  neuron_l3=neuron_k, neuron_l4=0,
                                                  window_size=window, n_features=num_features)

                    else:
                        print('Error: The network is not defined')
                        return None

                    # Define the compilation settings
                    # -------------------------------
                    model2.compile(optimizer='adam', loss=tf.keras.losses.MSE,
                                   metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])

                    # Fit the model with X_train and y_train
                    # --------------------------------------
                    history = model2.fit(X_train, y_train, epochs=200, validation_split=0.2, shuffle=True,
                                         batch_size=128, callbacks=[es_callback], verbose=0)

                    # Loop for every block
                    # --------------------
                    mae_num = 0
                    mse_num = 0
                    mae_den = 0
                    for block in range(0, itermax):
                        # Create windows of the block
                        X_testaux, y_testaux = create_windows_np(test_datanorm[block], window, horizon, shuffle=False)

                        # Evaluate the model
                        results = model2.evaluate(X_testaux, y_testaux, verbose=0)

                        # Save the results of the combination
                        # Construct the MAE
                        if block == 0:
                            mae_num = len(y_testaux) * sigma_target[block] * results[1]
                            mse_num = len(y_testaux) * sigma_target[block] * sigma_target[block] * results[0]
                            mae_den = len(y_testaux)

                        else:
                            mae_num = mae_num + len(y_testaux) * sigma_target[block] * results[1]
                            mse_num = mse_num + len(y_testaux) * sigma_target[block] * sigma_target[block] * results[0]
                            mae_den = mae_den + len(y_testaux)

                    # Save the results of the combination
                    mae = mae_num / mae_den
                    mse = mse_num / mae_den

                    neuron1.append(neuron_i)
                    neuron2.append(neuron_j)
                    neuron3.append(neuron_k)
                    test_mae.append(mae)
                    test_rmse.append(np.sqrt(mse))

                    # Show resulta
                    print('Neuron_l1 = {:3d}, Neuron_l2 = {:3d}, Neuron_l3 = {:3d} | '
                          'Test RMSE: {:.3f}, Test MAE = {:.3f}'.format(neuron_i, neuron_j,
                                                                        neuron_k, np.sqrt(mse), mae))

                    # Save the best model
                    if best_mae > mae:
                        best_model = model2
                        best_mae = mae

    # Show the Best case
    valor_min = min(test_mae)
    indice_min = test_mae.index(valor_min)

    print('\nThe best model was:')
    print('Neuron_l1 = {:3d}, Neuron_l2 = {:3d}, Neuron_l3 = {:3d} | Test RMSE: {:.3f}, Test MAE = {:.3f}'.format(
        neuron1[indice_min], neuron2[indice_min], neuron3[indice_min], test_rmse[indice_min], valor_min))

    # Return the best model
    return best_model


def tune_with_loop_multivariable(df, train_data, test_data, window, horizon, network, target):
    # Parameters settings
    target_col_idx = df.columns.get_loc(target)
    num_features = len(df.columns)

    neuron_l1 = [32, 64, 128, 256]
    neuron_l2 = [0, 32, 64, 128, 256]
    neuron_l3 = [0, 32, 64, 128, 256]

    # TREAT TRAIN DATASET
    # -------------------
    # Multivariable case
    # Normalize and save mu, sigma of the columns
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

    # Auxiliar variables
    neuron1 = []
    neuron2 = []
    neuron3 = []
    test_mae = []
    test_rmse = []

    # Model and mae variables
    best_mae = float('inf')
    best_model = None

    # Early Stopping Callback
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

    # Loop to know the best model
    print('Starting the tuning of the {} network'.format(network))
    print('-'*50)
    for neuron_j in neuron_l2:
        for neuron_i in neuron_l1:
            if neuron_j == 0:
                # Define the model
                # ----------------
                if network == 'LSTM':
                    model2 = create_deeplstmmodel(neuron_l1=neuron_i, neuron_l2=neuron_j, neuron_l3=0,
                                                  window_size=window, n_features=num_features)

                elif network == 'GRU':
                    model2 = create_deepgrumodel(neuron_l1=neuron_i, neuron_l2=neuron_j, neuron_l3=0, neuron_l4=0,
                                                 window_size=window, n_features=num_features)

                elif network == 'SimpleRNN':
                    model2 = create_simplernnmodel(neuron_l1=neuron_i, neuron_l2=neuron_j, neuron_l3=0, neuron_l4=0,
                                                   window_size=window, n_features=num_features)

                elif network == 'Convolutional':
                    model2 = create_deepconvmodel(filters_l1=neuron_i, filters_l2=neuron_j, filters_l3=0,
                                                  window_size=window, n_features=num_features)

                elif network == 'FeedForward':
                    model2 = create_deepmodel(neuron_l1=neuron_i, neuron_l2=neuron_j, neuron_l3=0, neuron_l4=0,
                                              window_size=window, n_features=num_features)

                else:
                    print('Error: The network is not defined')
                    return None

                # Define the compilation settings
                # -------------------------------
                model2.compile(optimizer='adam', loss=tf.keras.losses.MSE, metrics=['mae'])

                # Fit the model with X_train and y_train
                # --------------------------------------
                history = model2.fit(X_train, y_train, epochs=200, validation_split=0.2, shuffle=True,
                                     batch_size=128, callbacks=[es_callback], verbose=0)

                # Loop for every block
                # --------------------
                mae_num = 0
                mse_num = 0
                mae_den = 0
                for block in range(0, itermax):
                    # Create windows of the block
                    X_testaux, y_testaux = create_windows_multivariate_np(test_datanorm[block], window, horizon,
                                                                          target_col_idx=target_col_idx, shuffle=False)

                    # Evaluate the model
                    results = model2.evaluate(X_testaux, y_testaux, verbose=0)

                    # Construct the MAE
                    if block == 0:
                        mae_num = len(y_testaux) * sigma_target[block] * results[1]
                        mse_num = len(y_testaux) * sigma_target[block] * sigma_target[block] * results[0]
                        mae_den = len(y_testaux)

                    else:
                        mae_num = mae_num + len(y_testaux) * sigma_target[block] * results[1]
                        mse_num = mse_num + len(y_testaux) * sigma_target[block] * sigma_target[block] * results[0]
                        mae_den = mae_den + len(y_testaux)

                # Save the results of the combination
                mae = mae_num / mae_den
                mse = mse_num / mae_den

                neuron1.append(neuron_i)
                neuron2.append(neuron_j)
                neuron3.append(0)
                test_mae.append(mae)
                test_rmse.append(np.sqrt(mse))

                # Show the results
                print('Neuron_l1 = {:3d}, Neuron_l2 = {:3d}, Neuron_l3 = {:3d} | '
                      'Test RMSE: {:.3f}, Test MAE = {:.3f}'.format(neuron_i, neuron_j, 0, np.sqrt(mse), mae))

                # Save the best model
                if best_mae > mae:
                    best_model = model2
                    best_mae = mae

            else:
                for neuron_k in neuron_l3:
                    # Define the model
                    # ----------------
                    if network == 'LSTM':
                        model2 = create_deeplstmmodel(neuron_l1=neuron_i, neuron_l2=neuron_j, neuron_l3=neuron_k,
                                                      window_size=window, n_features=num_features)

                    elif network == 'GRU':
                        model2 = create_deepgrumodel(neuron_l1=neuron_i, neuron_l2=neuron_j,
                                                     neuron_l3=neuron_k, neuron_l4=0,
                                                     window_size=window, n_features=num_features)

                    elif network == 'SimpleRNN':
                        model2 = create_simplernnmodel(neuron_l1=neuron_i, neuron_l2=neuron_j,
                                                       neuron_l3=neuron_k, neuron_l4=0,
                                                       window_size=window, n_features=num_features)

                    elif network == 'Convolutional':
                        model2 = create_deepconvmodel(filters_l1=neuron_i, filters_l2=neuron_j, filters_l3=neuron_k,
                                                      window_size=window, n_features=num_features)

                    elif network == 'FeedForward':
                        model2 = create_deepmodel(neuron_l1=neuron_i, neuron_l2=neuron_j,
                                                  neuron_l3=neuron_k, neuron_l4=0,
                                                  window_size=window, n_features=num_features)

                    else:
                        print('Error: The network is not defined')
                        return None

                    # Define the compilation settings
                    # -------------------------------
                    model2.compile(optimizer='adam', loss=tf.keras.losses.MSE,
                                   metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])

                    # Fit the model with X_train and y_train
                    # --------------------------------------
                    history = model2.fit(X_train, y_train, epochs=200, validation_split=0.2, shuffle=True,
                                         batch_size=128, callbacks=[es_callback], verbose=0)

                    # Loop for every block
                    # --------------------
                    mae_num = 0
                    mse_num = 0
                    mae_den = 0
                    for block in range(0, itermax):
                        # Create windows of the block
                        X_testaux, y_testaux = create_windows_multivariate_np(test_datanorm[block], window, horizon,
                                                                              target_col_idx=target_col_idx,
                                                                              shuffle=False)
                        # Evaluate the model
                        results = model2.evaluate(X_testaux, y_testaux, verbose=0)

                        # Save the results of the combination
                        # Construct the MAE
                        if block == 0:
                            mae_num = len(y_testaux) * sigma_target[block] * results[1]
                            mse_num = len(y_testaux) * sigma_target[block] * sigma_target[block] * results[0]
                            mae_den = len(y_testaux)

                        else:
                            mae_num = mae_num + len(y_testaux) * sigma_target[block] * results[1]
                            mse_num = mse_num + len(y_testaux) * sigma_target[block] * sigma_target[block] * results[0]
                            mae_den = mae_den + len(y_testaux)

                    # Save the results of the combination
                    mae = mae_num / mae_den
                    mse = mse_num / mae_den

                    neuron1.append(neuron_i)
                    neuron2.append(neuron_j)
                    neuron3.append(neuron_k)
                    test_mae.append(mae)
                    test_rmse.append(np.sqrt(mse))

                    # Show resulta
                    print('Neuron_l1 = {:3d}, Neuron_l2 = {:3d}, Neuron_l3 = {:3d} | '
                          'Test RMSE: {:.3f}, Test MAE = {:.3f}'.format(neuron_i, neuron_j,
                                                                        neuron_k, np.sqrt(mse), mae))

                    # Save the best model
                    if best_mae > mae:
                        best_model = model2
                        best_mae = mae

    # Show the Best case
    valor_min = min(test_mae)
    indice_min = test_mae.index(valor_min)

    print('\nThe best model was:')
    print('Neuron_l1 = {:3d}, Neuron_l2 = {:3d}, Neuron_l3 = {:3d} | Test RMSE: {:.3f}, Test MAE = {:.3f}'.format(
        neuron1[indice_min], neuron2[indice_min], neuron3[indice_min], test_rmse[indice_min], valor_min))

    # Return the best model
    return best_model
