import pandas as pd
import numpy as np


def analyze_predictions(y_test, y_pred, network):
    # Texto de Inicio
    print('\nCalculating errores of the {} network'.format(network))
    print('-'*50)

    # Calculate the Results
    data = pd.DataFrame()
    data['y_test'] = y_test
    data['y_pred'] = y_pred
    data['delta'] = data['y_pred'] - data['y_test']
    data['deltaabs'] = np.abs(data['y_pred'] - data['y_test'])

    # Show the Info by Console
    print('Maximum difference {} es = {:.3f}'.format(network, max(data['deltaabs'])))
    print('MAE {} es = {:.3f}'.format(network, np.mean(data['deltaabs'])))
    print('RMSE {} es = {:.3f}\n'.format(network, np.mean(data['deltaabs'] * data['deltaabs']) ** 0.5))

    # Return data
    return data


def analyze_allnetworks(data, data2, data3, data4, data5):
    mae_vector = [np.mean(data['deltaabs']), np.mean(data2['deltaabs']), np.mean(data3['deltaabs']),
                  np.mean(data4['deltaabs']), np.mean(data5['deltaabs'])]
    rmse_vector = [np.mean(data['deltaabs'] * data['deltaabs']) ** 0.5,
                   np.mean(data2['deltaabs'] * data2['deltaabs']) ** 0.5,
                   np.mean(data3['deltaabs'] * data3['deltaabs']) ** 0.5,
                   np.mean(data4['deltaabs'] * data4['deltaabs']) ** 0.5,
                   np.mean(data5['deltaabs'] * data5['deltaabs']) ** 0.5]
    maxdiff_vector = [max(data['deltaabs']), max(data2['deltaabs']), max(data3['deltaabs']),
                      max(data4['deltaabs']), max(data5['deltaabs'])]

    names = ['Feed Forward', 'LSTM', 'GRU', 'Simple RNN', 'Convolutional']

    test_results = pd.DataFrame()

    test_results['Type'] = names
    test_results['RMSE'] = rmse_vector
    test_results['MAE'] = mae_vector
    test_results['MaxDiff'] = maxdiff_vector

    # Show the results
    print('See the differences between Networks')
    print('-'*40)
    print(test_results.sort_values(by="RMSE"))

    return test_results


def analyze_allnetworks_multi(data6, data7, data8, data9, data10):
    mae_vector = [np.mean(data6['deltaabs']), np.mean(data7['deltaabs']),
                  np.mean(data8['deltaabs']), np.mean(data9['deltaabs']),
                  np.mean(data10['deltaabs'])]

    rmse_vector = [np.mean(data6['deltaabs'] * data6['deltaabs']) ** 0.5,
                   np.mean(data7['deltaabs'] * data7['deltaabs']) ** 0.5,
                   np.mean(data8['deltaabs'] * data8['deltaabs']) ** 0.5,
                   np.mean(data9['deltaabs'] * data9['deltaabs']) ** 0.5,
                   np.mean(data10['deltaabs'] * data10['deltaabs']) ** 0.5]

    maxdiff_vector = [max(data6['deltaabs']), max(data7['deltaabs']), max(data8['deltaabs']),
                      max(data9['deltaabs']), max(data10['deltaabs'])]

    names = ['Importance 0.3', 'Importance 0.2', 'Importance 0.2 + fe', 'Importance 0.1', 'Importance 0.1 + fe']

    test_results = pd.DataFrame()

    test_results['Type'] = names
    test_results['RMSE'] = rmse_vector
    test_results['MAE'] = mae_vector
    test_results['MaxDiff'] = maxdiff_vector

    print('See the differences between Networks by RMSE')
    print('-' * 50)
    print(test_results.sort_values(by="RMSE"))

    print('\nSee the differences between Networks by MAE')
    print('-' * 50)
    print(test_results.sort_values(by="MAE"))

    print('\nSee the differences between Networks by MaxDiff')
    print('-' * 50)
    print(test_results.sort_values(by="MaxDiff"))

    return test_results
