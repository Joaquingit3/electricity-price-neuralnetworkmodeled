import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


def create_multigraph(y_test, y_pred, y_pred2, y_pred3, y_pred4, y_pred5,
                      test_data, window, horizon, target, save=True):
    # Create the date vector
    n_years = int(test_data.shape[0] / 365)
    resto_years = (test_data.shape[0] % 365)
    fechas = []

    if resto_years == 0:
        for i in range(1, n_years + 1):
            extrem_izq = (window + horizon - 1) + 365 * (i - 1)
            extrem_drch = 365 * i
            fechas_aux = test_data.index.date[extrem_izq:extrem_drch]
            fechas.append(fechas_aux)
    else:
        for i in range(1, n_years + 1):
            extrem_izq = (window + horizon - 1) + 365 * (i - 1)
            extrem_drch = 365 * i
            fechas_aux = test_data.index.date[extrem_izq:extrem_drch]
            fechas.append(fechas_aux)

        # Add the last extrem
        extrem_izq = (window + horizon - 1) + 365 * n_years
        fechas_aux = test_data.index.date[extrem_izq:]
        fechas.append(fechas_aux)

    # Create the fechas_extend
    fechas_extend = []
    for i in range(len(fechas)):
        # First iteration
        if i == 0:
            fechas_extend = list(fechas[i])
        else:
            fechas_extend += list(fechas[i])

    # PLOT
    # Define the fig size and gridspects
    fig = plt.figure(figsize=(16, 6))
    nrows = n_years + 1
    ncols = n_years + 1
    gs0 = gridspec.GridSpec(nrows, ncols, figure=fig)

    # Set style
    sns.set_style("darkgrid")

    # First graph
    # -----------
    last_tick = int(test_data.shape[0] / 100) * 100
    step = int(last_tick/5)
    vect_aux = [i*step for i in range(6)]
    index = [fechas_extend[i] for i in vect_aux]

    ax1 = fig.add_subplot(gs0[:, :-1])
    plt.plot(y_test)
    plt.plot(y_pred)
    plt.plot(y_pred2)
    plt.plot(y_pred3)
    plt.plot(y_pred4)
    plt.plot(y_pred5)

    # Graph the vertical lines
    for year in range(1, n_years + 1):
        extrem_x = float((365 - (window + horizon - 1)) * year - 0.5)
        ax1.axvline(x=extrem_x, ymin=-100, ymax=700, linestyle='--', color='grey')  # Vertical line

    ax1.legend([target, 'Deep FeedForward Predictions ', 'LSTM Predictions',
                'GRU Predictions', 'SimpleRNN Predictions', 'Convolutional Predictions'],
               loc='upper left', fontsize=11)
    ax1.set_xlabel('Date', fontsize=12, fontweight='light')

    if target == 'Spot_Price':
        ax1.set_ylabel('{} (€/MWh)'.format(target), fontsize=12, fontweight='light')
    else:
        ax1.set_ylabel('{}'.format(target), fontsize=12, fontweight='light')

    ax1.set_title('{} Predictions - window = {}, horizon = {}'.format(target, window, horizon), fontsize=14,
                  fontweight='bold', loc='left')

    ax1.set_xticks(np.arange(0, last_tick+step, step), index)
    ax1.tick_params(labelsize=11)

    # Divide the last space depending on the number of rows
    gs1 = gs0[:, -1].subgridspec(nrows, 1, hspace=0.4)

    # Second graph
    # ------------
    # Second graph - Block of one year maximum
    for i in range(len(fechas)):
        # Check the len of fechas[i]
        extrem_sup = int(len(fechas[i]) / 100) + 1
        extrem_sup = extrem_sup * 100

        # Calculate the fechas index of the block
        vect_aux = np.arange(0, extrem_sup, 100)
        fechas_aux = fechas[i]
        index = [fechas_aux[day] for day in vect_aux]

        # Plot the block series
        ax = fig.add_subplot(gs1[i])

        extrem_izq = (365 - (window + horizon - 1)) * i
        extrem_drch = (365 - (window + horizon - 1)) * (i + 1)

        ax.plot(y_test[extrem_izq:extrem_drch])
        ax.plot(y_pred[extrem_izq:extrem_drch])
        ax.plot(y_pred2[extrem_izq:extrem_drch])
        ax.plot(y_pred3[extrem_izq:extrem_drch])
        ax.plot(y_pred4[extrem_izq:extrem_drch])
        ax.plot(y_pred5[extrem_izq:extrem_drch])

        ax.set_title('{} Predictions - Year {}'.format(target, i + 1),
                     fontsize=10, fontweight='bold', loc='left')

        ax.set_xticks(np.arange(0, extrem_sup, 100), index)
        ax.tick_params(labelsize=11)

    # Grid of the graph
    plt.tight_layout()

    # Check the save parameter
    if save:
        plt.savefig('./Results/Graphs/All_univariable_withblocks.pdf', format='pdf')
    else:
        plt.show()


def create_multigraph_multivariante(y_test, y_pred_i03, y_pred_i02, y_pred_i02fe, y_pred_i01, y_pred_i01fe,
                                    test_data, window, horizon, target, save=True):
    # Create the date vector
    n_years = int(test_data.shape[0] / 365)
    resto_years = (test_data.shape[0] % 365)
    fechas = []

    if resto_years == 0:
        for i in range(1, n_years + 1):
            extrem_izq = (window + horizon - 1) + 365 * (i - 1)
            extrem_drch = 365 * i
            fechas_aux = test_data.index.date[extrem_izq:extrem_drch]
            fechas.append(fechas_aux)
    else:
        for i in range(1, n_years + 1):
            extrem_izq = (window + horizon - 1) + 365 * (i - 1)
            extrem_drch = 365 * i
            fechas_aux = test_data.index.date[extrem_izq:extrem_drch]
            fechas.append(fechas_aux)

        # Add the last extrem
        extrem_izq = (window + horizon - 1) + 365 * n_years
        fechas_aux = test_data.index.date[extrem_izq:]
        fechas.append(fechas_aux)

    # Create the fechas_extend
    fechas_extend = []
    for i in range(len(fechas)):
        # First iteration
        if i == 0:
            fechas_extend = list(fechas[i])
        else:
            fechas_extend += list(fechas[i])

    # PLOT
    # Define the fig size and gridspects
    fig = plt.figure(figsize=(16, 6))
    nrows = n_years + 1
    ncols = n_years + 1
    gs0 = gridspec.GridSpec(nrows, ncols, figure=fig)

    # Set style
    sns.set_style("darkgrid")

    # First graph
    # -----------
    last_tick = int(test_data.shape[0] / 100) * 100
    step = int(last_tick / 5)
    vect_aux = [i * step for i in range(6)]
    index = [fechas_extend[i] for i in vect_aux]

    ax1 = fig.add_subplot(gs0[:, :-1])

    ax1.plot(y_test)
    # ax1.plot(y_pred2)
    ax1.plot(y_pred_i03)
    ax1.plot(y_pred_i02)
    ax1.plot(y_pred_i02fe)
    ax1.plot(y_pred_i01)
    ax1.plot(y_pred_i01fe)
    # ax1.plot(y_pred_weather)

    # Graph the vertical lines
    for year in range(1, n_years + 1):
        extrem_x = float((365 - (window + horizon - 1)) * year - 0.5)
        ax1.axvline(x=extrem_x, ymin=-100, ymax=700, linestyle='--', color='grey')  # Vertical line

    ax1.legend([target, 'LSTM, Importance = 0.3', 'LSTM, Importance = 0.2',
                'LSTM, Importance = 0.2 + features',
                'LSTM, Importance = 0.1', 'LSTM, Importance = 0.1 + features'],
               loc='upper left', fontsize=11)

    ax1.set_xlabel('Date', fontsize=12, fontweight='light')

    if target == 'Spot_Price':
        ax1.set_ylabel('{} (€/MWh)'.format(target), fontsize=12, fontweight='light')
    else:
        ax1.set_ylabel('{}'.format(target), fontsize=12, fontweight='light')

    ax1.set_title('{} Predictions - window = {}, horizon = {}'.format(target, window, horizon), fontsize=14,
                  fontweight='bold', loc='left')

    ax1.set_xticks(np.arange(0, last_tick + step, step), index)
    ax1.tick_params(labelsize=11)

    # Divide the last space depending on the number of rows
    gs1 = gs0[:, -1].subgridspec(nrows, 1, hspace=0.4)

    # Second graph
    # ------------
    # Second graph - Block of one year maximum
    for i in range(len(fechas)):
        # Check the len of fechas[i]
        extrem_sup = int(len(fechas[i]) / 100) + 1
        extrem_sup = extrem_sup * 100

        # Calculate the fechas index of the block
        vect_aux = np.arange(0, extrem_sup, 100)
        fechas_aux = fechas[i]
        index = [fechas_aux[day] for day in vect_aux]

        # Plot the block series
        ax = fig.add_subplot(gs1[i])

        extrem_izq = (365 - (window + horizon - 1)) * i
        extrem_drch = (365 - (window + horizon - 1)) * (i + 1)

        ax.plot(y_test[extrem_izq:extrem_drch])
        # ax.plot(y_pred2[extrem_izq:extrem_drch])
        ax.plot(y_pred_i03[extrem_izq:extrem_drch])
        ax.plot(y_pred_i02[extrem_izq:extrem_drch])
        ax.plot(y_pred_i02fe[extrem_izq:extrem_drch])
        ax.plot(y_pred_i01[extrem_izq:extrem_drch])
        ax.plot(y_pred_i01fe[extrem_izq:extrem_drch])
        # ax.plot(y_pred_weather[extrem_izq:extrem_drch])

        ax.set_title('{} Predictions - Year {}'.format(target, i + 1),
                     fontsize=10, fontweight='bold', loc='left')

        ax.set_xticks(np.arange(0, extrem_sup, 100), index)
        ax.tick_params(labelsize=11)

    # Grid
    plt.tight_layout()

    # Check the save
    if save:
        plt.savefig('./Results/Graphs/All_multivariable_withblocks.pdf', format='pdf')
    else:
        plt.show()
