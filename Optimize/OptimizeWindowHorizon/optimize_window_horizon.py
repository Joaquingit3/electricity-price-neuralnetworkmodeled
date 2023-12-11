import keras
import tensorflow as tf
import numpy as np

from Functions.Dataset_Treatment.dataset_treatment import normalize_vector, divide_oneyearperiod
from Functions.Modeling.univariable_modeling import create_windows_np
from Functions.NeuralNetworks.neuralnetworks_functions import create_deepmodel


def optimize_window_horizon(train_data, test_data, windows, horizons):
    # Early stopping callback definition, stop the model when the validation metric can be improved
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

    # Auxiliar data
    window_aux = []
    horizon_aux = []
    test_rmse = []
    test_mae = []

    # Model and mae variables
    best_mae = float('inf')
    best_model = None

    # Parameters settings
    num_features = 1
    print('Starting the Window and Horizon optimization')
    print('-'*60)
    # Loop to window and horizon
    for window in windows:
        for horizon in horizons:
            # TREAT TRAIN DATASET
            # -------------------
            # Normalize and save mu, sigma of the columns
            train_datanorm = train_data.copy()
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

            # Define the model
            # ----------------
            model1 = create_deepmodel(neuron_l1=256, neuron_l2=256, neuron_l3=256, neuron_l4=0,
                                      window_size=window, n_features=num_features)

            # Define the compilation settings
            # -------------------------------
            model1.compile(optimizer='adam', loss=tf.keras.losses.MSE, metrics=['mae'])

            # Fit the model with X_train and y_train
            # --------------------------------------
            history = model1.fit(X_train, y_train, epochs=200, validation_split=0.2, shuffle=True,
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
                results = model1.evaluate(X_testaux, y_testaux, verbose=0)

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

            # Print results
            print('Window = {:2d}, horizon = {:3d} | Test RMSE: {:.3f}, Test MAE = {:.3f}'.format(window, horizon,
                                                                                                  np.sqrt(mse), mae))

            # Save the best combination
            window_aux.append(window)
            horizon_aux.append(horizon)
            test_mae.append(mae)
            test_rmse.append(np.sqrt(mse))

            # Save the best model
            if best_mae > mae:
                best_model = model1
                best_mae = mae

        # Next line from different window
        print('\n')

    # Show the Best case
    valor_min = min(test_mae)
    indice_min = test_mae.index(valor_min)

    print('The best model was:\n')
    print('Window = {:2d}, horizon = {:3d} | Test RMSE: {:.3f}, Test MAE = {:.3f}'.format(window_aux[indice_min],
                                                                                          horizon_aux[indice_min],
                                                                                          test_rmse[indice_min],
                                                                                          valor_min))
    # Save the best window and horizon
    window_best = window_aux[indice_min]
    horizon_best = horizon_aux[indice_min]

    # Return de los mejores modelos
    return window_best, horizon_best, test_mae, test_rmse, best_model
