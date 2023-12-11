import numpy as np
from pickle import dump

from Functions.Dataset_Treatment.dataset_treatment import import_dataset, split_train_test

from Plotter.OptimizeGraph.optimize_windowhorizon_graph import graph_optimize_window_horizon
from Plotter.Predictions.PredictionsGraphs import graph_singlepredictions_withblocks
from Plotter.Basic.basic_graphs import graph_train_test_split, graph_train_test_normalize, graph_column
from Plotter.MultiPredictions.multipredictions import create_multigraph, create_multigraph_multivariante

from Optimize.OptimizeWindowHorizon.optimize_window_horizon import optimize_window_horizon
from Optimize.OptimizeUnivariable.optimize_univariable import (generate_predictions_univariable,
                                                               generate_predictions_lstm_univariable,
                                                               generate_predictions_gru_univariable,
                                                               generate_predictions_simplernn_univariable,
                                                               generate_predictions_conv_univariable)

from Optimize.OptimizeMultivariable.optimize_multivariable import (fit_and_generate_predictions_multivariable,
                                                                   generate_predictions_multivariable)

from Analyze.Analyze_functions import analyze_predictions, analyze_allnetworks, analyze_allnetworks_multi

from Tunning.Loop_Tunning import tune_with_loop_univariable, tune_with_loop_multivariable


def main_univariable():
    # PARAMETERS OF THE PROCESSES
    # ---------------------------
    optimize_wh = True
    proof_othernetworks = True
    use_lstm = True
    use_gru = True
    use_simplernn = True
    use_conv = True
    graph_multigraph = True

    # 1) Optimizar (window, horizon) con univariable y DeepModel
    # 1.1) Importar la data
    data_location = "./Data/df_weather_all.csv"
    target = 'Spot_Price'

    df = import_dataset(data_location)
    graph_column(df, column=target, save=True)

    # 1.2) Generar el df univariable
    df_uni = df[target].copy()

    # 1.3) Train-test split
    train_data, test_data = split_train_test(df_uni, split_fraction=0.7)
    graph_train_test_split(train_data, test_data, target=target, save=True)
    graph_train_test_normalize(test_data, target=target, save=True)

    if optimize_wh:
        # Network parameter
        network = 'FeedForward'

        # 1.4) Start the optimizer
        # Window and horizon vector
        windows = [7, 16, 19, 22, 30, 33]
        horizons = [1, 7, 14, 15, 30, 60, 90, 120, 180]

        window, horizon, test_mae, test_rmse, model1 = optimize_window_horizon(train_data, test_data, windows, horizons)

        # 1.5) Generate optimization graph
        graph_optimize_window_horizon(test_mae, test_rmse, windows, horizons, save=True)

        # 1.6) Generate predictions with the best_model found
        y_pred, y_test = generate_predictions_univariable(window, horizon, test_data, model1)

        # 1.7) Graph the predictions of the best case
        graph_singlepredictions_withblocks(test_data, y_test, y_pred, window, horizon,
                                           target=target, network=network, save=True)

        # 1.8) Check the test results of the best model
        data = analyze_predictions(y_test, y_pred, network)

        # 1.8) Save the predictions and the model
        save_path = './Results/Predictions/y_pred_' + network + '.csv'
        np.savetxt(save_path, y_pred, delimiter=';', fmt='%.8f')

        # 1.9) Save the model
        save_path = './Results/Models/2023_' + network + '_Univariable.pkl'
        dump(model1, open(save_path, 'wb'))

    if proof_othernetworks:
        # Check if the variable window is defined
        if "window" not in globals():
            window = 7

        # Check if the variable horizon is defined
        if "horizon" not in globals():
            horizon = 1

        # LSTM
        # -------------------------------------------------
        if use_lstm:
            # Network parameter
            network = 'LSTM'

            # Probar con el modelo LSTM generando predicciones
            y_pred2, y_test, model2 = generate_predictions_lstm_univariable(train_data, test_data, window, horizon)

            # Graficar las predicciones
            graph_singlepredictions_withblocks(test_data, y_test, y_pred2, window, horizon,
                                               target=target, network=network, save=True)

            # Check the test results
            data2 = analyze_predictions(y_test, y_pred2, network)

            # Save the predictions
            save_path = './Results/Predictions/y_pred_' + network + '.csv'
            np.savetxt(save_path, y_pred2, delimiter=';', fmt='%.8f')

            # Save the model
            save_path = './Results/Models/2023_' + network + '_Univariable.pkl'
            dump(model2, open(save_path, 'wb'))

        # GRU
        # -------------------------------------------------
        if use_gru:
            # Network parameter
            network = 'GRU'

            # Probar con el modelo GRU generando predicciones
            y_pred3, y_test, model3 = generate_predictions_gru_univariable(train_data, test_data, window, horizon)

            # Graficar las predicciones
            graph_singlepredictions_withblocks(test_data, y_test, y_pred3, window, horizon,
                                               target=target, network=network, save=True)

            # Check the Test results
            data3 = analyze_predictions(y_test, y_pred3, network)

            # Save the predictions
            save_path = './Results/Predictions/y_pred_' + network + '.csv'
            np.savetxt(save_path, y_pred3, delimiter=';', fmt='%.8f')

            # Save the model
            save_path = './Results/Models/2023_' + network + '_Univariable.pkl'
            dump(model3, open(save_path, 'wb'))

        # SimpleRNN
        # -------------------------------------------------
        if use_simplernn:
            # Network parameter
            network = 'SimpleRNN'

            # Probar con el modelo SimpleRNN generando predicciones
            y_pred4, y_test, model4 = generate_predictions_simplernn_univariable(train_data, test_data, window, horizon)

            # Graficar las predicciones
            graph_singlepredictions_withblocks(test_data, y_test, y_pred4, window, horizon,
                                               target=target, network=network, save=True)

            # Check Test results
            data4 = analyze_predictions(y_test, y_pred4, network)

            # Save the predictions
            save_path = './Results/Predictions/y_pred_' + network + '.csv'
            np.savetxt(save_path, y_pred4, delimiter=';', fmt='%.8f')

            # Save the model
            save_path = './Results/Models/2023_' + network + '_Univariable.pkl'
            dump(model4, open(save_path, 'wb'))

        # Convolutional
        # -------------------------------------------------
        if use_conv:
            # Network parameter
            network = 'Convolutional'

            # Probar con el modelo SimpleRNN generando predicciones
            y_pred5, y_test, model5 = generate_predictions_conv_univariable(train_data, test_data, window, horizon)

            # Graficar las predicciones
            graph_singlepredictions_withblocks(test_data, y_test, y_pred5, window, horizon,
                                               target=target, network=network, save=True)

            # Check Test results
            data5 = analyze_predictions(y_test, y_pred5, network)

            # Save the predictions
            save_path = './Results/Predictions/y_pred_' + network + '.csv'
            np.savetxt(save_path, y_pred5, delimiter=';', fmt='%.8f')

            # Save the model
            save_path = './Results/Models/2023_' + network + '_Univariable.pkl'
            dump(model5, open(save_path, 'wb'))

        if graph_multigraph:
            # Crear el Gráfico Conjunto
            create_multigraph(y_test, y_pred, y_pred2, y_pred3, y_pred4, y_pred5, test_data, window, horizon,
                              target=target, save=True)

            # Crear el análisis multi de todas las redes
            test_results = analyze_allnetworks(data, data2, data3, data4, data5)

            # Save Test results
            save_path = './Results/TestResults/test_results_univariate.csv'

            # Guardar el array en un archivo CSV
            test_results.to_csv(save_path, header=True, index=False)


def main_multivariable():
    # Parameters
    multi_i03 = False
    multi_i02fe = False
    multi_i02 = False
    multi_i01 = False
    multi_i01fe = False
    multi_weatherall = True
    graph_multigraph = False

    # Check if the variable window is defined
    if "window" not in globals():
        window = 7

    # Check if the variable horizon is defined
    if "horizon" not in globals():
        horizon = 1

    target = 'Spot_Price'

    if multi_i03:
        # 1) Load the df
        data_location = "./Data/df_importance0.3.csv"
        network = 'LSTM'
        network_data = 'LSTM, Importance 0.3'
        save_data = 'LSTM_i03'

        df = import_dataset(data_location)

        # 2) Split train_test
        train_data, test_data = split_train_test(df, split_fraction=0.7)

        # 3) Generate predictions
        y_pred_i03, y_test, model6 = fit_and_generate_predictions_multivariable(df, train_data, test_data,
                                                                                window, horizon, network=network,
                                                                                target=target)

        # 4) Generate graph
        graph_singlepredictions_withblocks(test_data, y_test, y_pred_i03, window, horizon, network_data,
                                           target=target, save=True)

        # 5) Show the Test results
        data6 = analyze_predictions(y_test, y_pred_i03, network_data)

        # 6) Save predictions
        save_path = './Results/Predictions/y_pred_' + save_data + '.csv'
        np.savetxt(save_path, y_pred_i03, delimiter=';', fmt='%.8f')

        # 7) Save the model
        save_path = './Results/Models/2023_' + save_data + '_Multivariable.pkl'
        dump(model6, open(save_path, 'wb'))

    if multi_i02:
        # 1) Load the df
        data_location = "./Data/df_electrical_importance0.2.csv"
        network = 'LSTM'
        network_data = 'LSTM, Importance 0.2'
        save_data = 'LSTM_i02'

        df = import_dataset(data_location)

        # 2) Split train_test
        train_data, test_data = split_train_test(df, split_fraction=0.7)

        # 3) Generate predictions
        y_pred_i02, y_test, model7 = fit_and_generate_predictions_multivariable(df, train_data, test_data,
                                                                                window, horizon, network=network,
                                                                                target=target)

        # 4) Generate graph
        graph_singlepredictions_withblocks(test_data, y_test, y_pred_i02, window, horizon, network_data,
                                           target=target, save=True)

        # 5) Show the Test results
        data7 = analyze_predictions(y_test, y_pred_i02, network_data)

        # 6) Save predictions
        save_path = './Results/Predictions/y_pred_' + save_data + '.csv'
        np.savetxt(save_path, y_pred_i02, delimiter=';', fmt='%.8f')

        # 7) Save the model
        save_path = './Results/Models/2023_' + save_data + '_Multivariable.pkl'
        dump(model7, open(save_path, 'wb'))

    if multi_i02fe:
        # 1) Load the df
        data_location = "./Data/df_electricalandfeature_importance0.2.csv"
        network = 'LSTM'
        network_data = 'LSTM, Importance 0.2 + features'
        save_data = 'LSTM_i02fe'

        df = import_dataset(data_location)

        # 2) Split train_test
        train_data, test_data = split_train_test(df, split_fraction=0.7)

        # 3) Generate predictions
        y_pred_i02fe, y_test, model8 = fit_and_generate_predictions_multivariable(df, train_data, test_data,
                                                                                  window, horizon, network=network,
                                                                                  target=target)

        # 4) Generate graph
        graph_singlepredictions_withblocks(test_data, y_test, y_pred_i02fe, window, horizon,
                                           network_data, target=target, save=True)

        # 5) Show the Test results
        data8 = analyze_predictions(y_test, y_pred_i02fe, network_data)

        # 6) Save predictions
        save_path = './Results/Predictions/y_pred_' + save_data + '.csv'
        np.savetxt(save_path, y_pred_i02fe, delimiter=';', fmt='%.8f')

        # 7) Save the model
        save_path = './Results/Models/2023_' + save_data + '_Multivariable.pkl'
        dump(model8, open(save_path, 'wb'))

    if multi_i01:
        # 1) Load the df
        data_location = "./Data/df_electrical_importance0.1.csv"
        network = 'LSTM'
        network_data = 'LSTM, Importance 0.1'
        save_data = 'LSTM_i01'

        df = import_dataset(data_location)

        # 2) Split train_test
        train_data, test_data = split_train_test(df, split_fraction=0.7)

        # 3) Generate predictions
        y_pred_i01, y_test, model9 = fit_and_generate_predictions_multivariable(df, train_data, test_data,
                                                                                window, horizon, network=network,
                                                                                target=target)

        # 4) Generate graph
        graph_singlepredictions_withblocks(test_data, y_test, y_pred_i01, window, horizon, network_data,
                                           target=target, save=True)

        # 5) Show the Test results
        data9 = analyze_predictions(y_test, y_pred_i01, network_data)

        # 6) Save predictions
        save_path = './Results/Predictions/y_pred_' + save_data + '.csv'
        np.savetxt(save_path, y_pred_i01, delimiter=';', fmt='%.8f')

        # 7) Save the model
        save_path = './Results/Models/2023_' + save_data + '_Multivariable.pkl'
        dump(model9, open(save_path, 'wb'))

    if multi_i01fe:
        # 1) Load the df
        data_location = "./Data/df_electricalandfeature_importance0.1.csv"
        network = 'LSTM'
        network_data = 'LSTM, Importance 0.1 + features'
        save_data = 'LSTM_i01fe'

        df = import_dataset(data_location)

        # 2) Split train_test
        train_data, test_data = split_train_test(df, split_fraction=0.7)

        # 3) Generate predictions
        y_pred_i01fe, y_test, model10 = fit_and_generate_predictions_multivariable(df, train_data, test_data,
                                                                                   window, horizon, network=network,
                                                                                   target=target)

        # 4) Generate graph
        graph_singlepredictions_withblocks(test_data, y_test, y_pred_i01fe, window, horizon, network_data,
                                           target=target, save=True)

        # 5) Show the Test results
        data10 = analyze_predictions(y_test, y_pred_i01fe, network_data)

        # 6) Save predictions
        save_path = './Results/Predictions/y_pred_' + save_data + '.csv'
        np.savetxt(save_path, y_pred_i01fe, delimiter=';', fmt='%.8f')

        # 7) Save the model
        save_path = './Results/Models/2023_' + save_data + '_Multivariable.pkl'
        dump(model10, open(save_path, 'wb'))

    if multi_weatherall:
        # 1) Load the df
        data_location = "./Data/df_weather_all.csv"
        network = 'LSTM'
        network_data = 'LSTM, Weather All'
        save_data = 'LSTM_weather_all'

        df = import_dataset(data_location)

        # 2) Split train_test
        train_data, test_data = split_train_test(df, split_fraction=0.7)

        # 3) Generate predictions
        y_pred_weather_all, y_test, model11 = fit_and_generate_predictions_multivariable(df, train_data, test_data,
                                                                                         window, horizon, network=network,
                                                                                         target=target)

        # 4) Generate graph
        graph_singlepredictions_withblocks(test_data, y_test, y_pred_weather_all, window, horizon, network_data,
                                           target=target, save=True)

        # 5) Show the Test results
        data10 = analyze_predictions(y_test, y_pred_weather_all, network_data)

        # 6) Save predictions
        save_path = './Results/Predictions/y_pred_' + save_data + '.csv'
        np.savetxt(save_path, y_pred_weather_all, delimiter=';', fmt='%.8f')

        # 7) Save the model
        save_path = './Results/Models/2023_' + save_data + '_Multivariable.pkl'
        dump(model11, open(save_path, 'wb'))


    if graph_multigraph:
        # Create Joined Graph
        create_multigraph_multivariante(y_test, y_pred_i03, y_pred_i02, y_pred_i02fe, y_pred_i01, y_pred_i01fe,
                                        test_data, window, horizon, target=target, save=True)

        # Create the multi network error analysis
        test_results_multi = analyze_allnetworks_multi(data6, data7, data8, data9, data10)

        # Save Test results
        save_path = './Results/TestResults/test_results_multivariable.csv'
        test_results_multi.to_csv(save_path, header=True, index=False)


def main_tune():
    tune_univariable = True
    tune_multivariable = False

    if tune_univariable:
        # Parameters
        target = 'Spot_Price'
        network = 'LSTM'
        window = 7
        horizon = 1

        # 1) Load df
        data_location = "./Data/df_weather_all.csv"
        df = import_dataset(data_location)

        # 2) Reduce to univariable
        df_uni = df[target].copy()

        # 3) Train-Test Split
        train_data, test_data = split_train_test(df_uni, split_fraction=0.7)

        # 4) Tune the network
        best_model = tune_with_loop_univariable(train_data, test_data, window, horizon, network)

        # 5) Obtain the predictions
        y_predtune, y_test = generate_predictions_univariable(window, horizon, test_data, best_model)

        # 6) Graph predictions
        graph_singlepredictions_withblocks(test_data, y_test, y_predtune, window, horizon,
                                           network, target=target, save=True)

        # 7) Calculate the test_results
        datatune = analyze_predictions(y_test, y_predtune, network)

        # 8) Save the predictions
        save_path = './Results/Predictions/y_predtune_' + network + '.csv'
        np.savetxt(save_path, y_predtune, delimiter=';', fmt='%.8f')

        # 9) Save the model
        save_path = './Results/Models/2023_' + network + '_Univariabletuned.pkl'
        dump(best_model, open(save_path, 'wb'))

        # 10) Save the predictions
        save_path = './Results/TestResults/test_results_' + network + 'tuneunivariable.csv'
        datatune.to_csv(save_path, header=True, index=False)

    if tune_multivariable:
        # Parameters
        df_name = 'i03'
        target = 'Spot_Price'
        network = 'LSTM'
        save_data = network + '_' + df_name

        window = 7
        horizon = 1

        # 1) Load df
        if df_name == 'i03':
            data_location = "./Data/df_importance0.3.csv"
            df = import_dataset(data_location)

        elif df_name == 'i02':
            data_location = "./Data/df_electrical_importance0.2.csv"
            df = import_dataset(data_location)

        elif df_name == 'i02fe':
            data_location = "./Data/df_electricalandfeature_importance0.2.csv"
            df = import_dataset(data_location)

        elif df_name == 'i01':
            data_location = "./Data/df_electrical_importance0.1.csv"
            df = import_dataset(data_location)

        elif df_name == 'i01fe':
            data_location = "./Data/df_electricalandfeature_importance0.2.csv"
            df = import_dataset(data_location)

        elif df_name == 'WEATHER':
            data_location = "./Data/df_weather_all.csv"
            df = import_dataset(data_location)

        else:
            print('Error, df weather is not in the directory')

        # 2) Train-Test Split
        train_data, test_data = split_train_test(df, split_fraction=0.7)

        # 3) Tune the model
        best_model = tune_with_loop_multivariable(df, train_data, test_data, window, horizon, network, target)

        # 4) Obtain the predictions
        y_predtune, y_test = generate_predictions_multivariable(df, test_data, window, horizon, best_model, network)

        # 5) Graph predictions
        graph_singlepredictions_withblocks(test_data, y_test, y_predtune, window, horizon, network,
                                           target=target, save=False)

        # 6) Show test results
        data_tune = analyze_predictions(y_test, y_predtune, network=network)

        # 7) Save predictions
        save_path = './Results/Predictions/y_predtune_' + save_data + '_.csv'
        np.savetxt(save_path, y_predtune, delimiter=';', fmt='%.8f')

        # 8) Save test results
        save_path = './Results/TestResults/test_results_' + save_data + 'tuneMultivariable.csv'
        data_tune.to_csv(save_path, header=True, index=False)

        # 9) Save model
        save_path = './Results/Models/2023_' + save_data + '_MultivariableTuned.pkl'
        dump(best_model, open(save_path, 'wb'))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Parameters
    check_univariable = False
    check_multivariable = True
    tune_model = False

    if check_univariable:
        main_univariable()

    if check_multivariable:
        main_multivariable()

    if tune_model:
        main_tune()
