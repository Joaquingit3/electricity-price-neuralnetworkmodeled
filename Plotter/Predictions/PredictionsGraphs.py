import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


def graph_singlepredictions_withblocks(test_data, y_test, y_pred, window, horizon, network, target, save=True):
    # Create the date vector
    n_years = int(test_data.shape[0] / 365)
    resto_years = (test_data.shape[0] % 365)
    fechas = []

    if resto_years == 0:
        for i in range(1, n_years + 1):
            extrem_izq = (window + horizon - 1) + 365*(i-1)
            extrem_drch = 365 * i
            fechas_aux = test_data.index.date[extrem_izq:extrem_drch]
            fechas.append(fechas_aux)
    else:
        for i in range(1, n_years + 1):
            extrem_izq = (window + horizon - 1) + 365*(i-1)
            extrem_drch = 365 * i
            fechas_aux = test_data.index.date[extrem_izq:extrem_drch]
            fechas.append(fechas_aux)

        # Add the last extrem
        extrem_izq = (window + horizon - 1) + 365*n_years
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
    # Figure settings
    fig = plt.figure(figsize=(16, 6))
    nrows = n_years + 1
    ncols = n_years + 1
    gs0 = gridspec.GridSpec(nrows, ncols, figure=fig)

    # Style of the graph
    sns.set_style("darkgrid")

    # First graph
    # -----------
    last_tick = int(test_data.shape[0] / 100) * 100
    step = int(last_tick / 5)
    vect_aux = [i * step for i in range(6)]
    index = [fechas_extend[i] for i in vect_aux]

    ax1 = fig.add_subplot(gs0[:, :-1])
    ax1.plot(y_test)
    ax1.plot(y_pred)

    # Graph the vertical lines
    for year in range(1, n_years+1):
        extrem_x = float((365 - (window + horizon - 1)) * year - 0.5)
        ax1.axvline(x=extrem_x, ymin=-100, ymax=700, linestyle='--', color='grey')  # Vertical line

    predictions_text = 'Predictions ' + network
    ax1.legend([target, predictions_text], loc='upper left')
    ax1.set_xlabel('Date', fontsize=10, fontweight='light')

    if target == 'Spot_Price':
        ax1.set_ylabel('{} (€/MWh)'.format(target), fontsize=12, fontweight='light')
    else:
        ax1.set_ylabel('{}'.format(target), fontsize=12, fontweight='light')

    ax1.set_title('{} Predictions - window = {}, horizon = {}'.format(target, window, horizon),
                  fontsize=14, fontweight='bold', loc='left')

    ax1.set_xticks(np.arange(0, last_tick + step, step), index)
    ax1.tick_params(labelsize=11)

    # Divide the last space depending on the number of rows
    gs1 = gs0[:, -1].subgridspec(nrows, 1, hspace=0.4)

    # Second graph - Block of one year maximum
    for i in range(len(fechas)):
        # Define the extrem of the interval
        extrem_sup = int(len(fechas[i])/100) + 1
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

        ax.set_title('{} Prections - Year {}'.format(target, i + 1),
                     fontsize=10, fontweight='bold', loc='left')
        ax.set_xticks(np.arange(0, extrem_sup, 100), index)

    # Grid and show the graph
    plt.tight_layout()

    # Save or show the graph
    if save:
        titlefig = network + '_univariable_withblocks.pdf'
        path = './Results/Graphs/' + titlefig
        plt.savefig(path, format='pdf')
    else:
        plt.show()
