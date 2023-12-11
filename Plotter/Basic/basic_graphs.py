import matplotlib.pyplot as plt
import seaborn as sns

from Functions.Dataset_Treatment.dataset_treatment import normalize_vector, divide_oneyearperiod
from Functions.Modeling.univariable_modeling import create_windows_np


def graph_column(df, column, save=True):
    # Style
    sns.set_style("darkgrid")

    # Plot of the column
    plt.figure(figsize=(12, 4))
    plt.plot(df[column], label=column)

    # Axis and title
    plt.xlabel('Date', fontsize=12, fontweight='light')

    if column == 'Spot_Price':
        # We know the dimensions of the Spot_price
        plt.ylabel('{} (€/MWh)'.format(column), fontsize=12, fontweight='light')
    else:
        plt.ylabel('{}'.format(column), fontsize=12, fontweight='light')

    plt.title('{} Evolution'.format(column), fontsize=20, fontweight='bold', loc='left')
    plt.tick_params(labelsize=12)

    # Grid
    plt.tight_layout()

    # Check save
    if save:
        plt.savefig('./Results/Graphs/PrecioSpot.pdf', format='pdf')
    else:
        plt.show()


def graph_train_test_split(train_data, test_data, target, save=True):
    # Plot the DF train y test
    sns.set_style("darkgrid")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [8, 15]})
    fig.suptitle('Data split', horizontalalignment='left', x=0.125, y=1, fontsize=20, fontweight='bold')

    # First Graph
    ax1.plot(train_data, 'red')
    ax1.set_title('Train Data', fontsize=14, fontweight='regular', loc='left')
    ax1.set_xlabel('Date', fontsize=12, fontweight='light')

    if target == 'Spot_Price':
        ax1.set_ylabel('{} (€/MWh)'.format(target), fontsize=12, fontweight='light')
    else:
        ax1.set_ylabel('{}'.format(target), fontsize=12, fontweight='light')

    ax1.tick_params(labelsize=11)

    # Second Graph
    ax2.plot(test_data)
    ax2.set_title('Test Data', fontsize=14, fontweight='regular', loc='left')
    ax2.set_xlabel('Date', fontsize=12, fontweight='light')

    if target == 'Spot_Price':
        ax2.set_ylabel('{} (€/MWh)'.format(target), fontsize=12, fontweight='light')
    else:
        ax2.set_ylabel('{}'.format(target), fontsize=12, fontweight='light')

    ax2.tick_params(labelsize=11)

    # Check the save parameter
    if save:
        plt.savefig('./Results/Graphs/TrainTestSplit.pdf', format='pdf')
    else:
        plt.show()


def graph_train_test_normalize(test_data, target, save):
    # Create the blows
    X_test, y_test = create_windows_np(test_data, 1, 0, shuffle=False)

    # Divide in years blocks
    y_test_divide, itermax = divide_oneyearperiod(y_test)

    # Normalizamos cada bloque
    y1_norm, mu1, sigma1 = normalize_vector(y_test_divide[0])
    y2_norm, mu2, sigma2 = normalize_vector(y_test_divide[1])
    y3_norm, mu3, sigma3 = normalize_vector(y_test_divide[2])

    # Results
    print('Metrics before')
    print('mu1 = {}, sigma1 = {}'.format(mu1, sigma1))
    print('mu2 = {}, sigma2 = {}'.format(mu2, sigma2))
    print('mu3 = {}, sigma3 = {}\n'.format(mu3, sigma3))

    print('Metrics after')
    print('mu1 = {}, sigma1 = {}'.format(y1_norm.mean(), y1_norm.std()))
    print('mu2 = {}, sigma2 = {}'.format(y2_norm.mean(), y2_norm.std()))
    print('mu3 = {}, sigma3 = {}\n'.format(y3_norm.mean(), y3_norm.std()))

    # Plot
    # ---------------------------------------------------------------------------
    # Style
    sns.set_style("darkgrid")

    # Graph define
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle('Test division', horizontalalignment='left', x=0.125, y=1, fontsize=20, fontweight='bold')

    # First block
    ax1.plot(y1_norm, 'red')
    ax1.set_title('First year', loc='left', fontsize=14)
    ax1.set_xlabel('Date', fontsize=12, fontweight='light')
    ax1.set_ylabel('Normalize {}'.format(target), fontsize=12, fontweight='light')
    ax1.tick_params(labelsize=11)
    ax1.legend(['µ1 = {:.3f}\nσ1 = {:.3f}'.format(mu1, sigma1)], loc='upper left')

    # Second Block
    ax2.plot(y2_norm, 'orange')
    ax2.set_title('Second year', loc='left', fontsize=14)
    ax2.set_xlabel('Date', fontsize=12, fontweight='light')
    ax2.set_ylabel('Normalize'.format(target), fontsize=12, fontweight='light')
    ax2.tick_params(labelsize=11)
    ax2.legend(['µ2 = {:.3f}\nσ2 = {:.3f}'.format(mu2, sigma2)], loc='upper left')

    # Third block
    ax3.plot(y3_norm, 'blue')
    ax3.set_title('Third year', loc='left', fontsize=14)
    ax3.set_xlabel('Dare', fontsize=12, fontweight='light')
    ax3.set_ylabel('Normalize {}'.format(target), fontsize=12, fontweight='light')
    ax3.tick_params(labelsize=11)
    ax3.legend(['µ3 = {:.3f}\nσ3 = {:.3f}'.format(mu3, sigma3)], loc='upper right')

    # Check the save
    if save:
        plt.savefig('./Results/Graphs/BlockTestNorm.pdf', format='pdf')
    else:
        plt.show()
