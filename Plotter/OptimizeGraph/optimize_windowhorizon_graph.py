import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def graph_optimize_window_horizon(test_mae, test_rmse, windows, horizons, save=True):
    # WINDOW - HORIZON GRAPH ANALYSIS
    # Define the style
    sns.set_style("darkgrid")

    # Define the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # TEST MAE
    # --------
    vector = np.array(test_mae)

    # Divide vector in 6 elements
    segmentos = np.split(vector, 6)

    for i, segmento in enumerate(segmentos):
        ax1.plot(np.arange(1, 10), segmento, label=f'Window = {windows[i]}')

    ax1.set_xlabel('Horizon')
    ax1.set_ylabel('Test MAE')
    ax1.set_title('Test MAE vs (window, horizon)', loc='left', fontweight='bold')
    ax1.set_xticks(np.arange(1, 10), horizons)
    ax1.legend()

    # TEST RMSE
    vector = np.array(test_rmse)

    # Dividir el vector en segmentos de 10 elementos
    segmentos = np.split(vector, 6)

    for i, segmento in enumerate(segmentos):
        ax2.plot(np.arange(1, 10), segmento, label=f'Window = {windows[i]}')

    ax2.set_xlabel('Horizon')
    ax2.set_ylabel('Test RMSE')
    ax2.set_title('Test RMSE vs (window, horizon)', loc='left', fontweight='bold')
    ax2.set_xticks(np.arange(1, 10), horizons)
    ax2.legend()

    # Save or show the graph
    if save:
        plt.savefig('./Results/Graphs/Window_horizon_optimization.pdf', format='pdf')
    else:
        plt.show()
