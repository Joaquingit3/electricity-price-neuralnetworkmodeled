import tensorflow as tf
import keras


# Function to create a Deep model
def create_deepmodel(neuron_l1, neuron_l2, neuron_l3, neuron_l4, window_size, n_features):
    """
        This functions creates a Deep Neural Network

        Input:
        - neuron_l1 (int): number of neurons desire in the first layer, we check
                     if this number is zero in that case we show an error message
        - neuron_l2 (int): number of neurons desire in the second layer, if this number
                     is zero we don't add the second layer
        - neuron_l3 (int): number of neurons desire in the third layer, if this number
                     is zero we don't add the third layer
        - neuron_l4 (int): number of neurons desire in the fourth layer, if this number
                     is zero we don't add the fourth layer
        - window_size (int): contains the dimension of the window
        - n_features (int): contains the number of features

        Output
        - model (keras.Model): The neural network
    """
    # Create the network
    # Input layer
    inputs = tf.keras.layers.Input(shape=(window_size, n_features))

    # Flatten layer, necessary to feedforward
    flatten_layer = tf.keras.layers.Flatten()(inputs)

    # Hidden layers
    if neuron_l1 == 0:
        print("Error - First Layer (l1) can't have 0 neurons")

    else:
        # Hidden layer 1
        hidd_1 = tf.keras.layers.Dense(units=neuron_l1, activation='relu')(flatten_layer)

        # Check the second layer
        if neuron_l2 == 0:
            # Output layer
            outputs = tf.keras.layers.Dense(units=1, activation='linear')(hidd_1)

            # Define the network
            model = tf.keras.Model(inputs=inputs, outputs=outputs)

            # Return the model
            return model
        else:
            # Hidden layer 2
            hidd_2 = tf.keras.layers.Dense(units=neuron_l2, activation='relu')(hidd_1)

            # Check the third layer
            if neuron_l3 == 0:
                # Output layer
                outputs = tf.keras.layers.Dense(units=1, activation='linear')(hidd_2)

                # Define the network
                model = tf.keras.Model(inputs=inputs, outputs=outputs)

                # Return the model
                return model

            else:
                # Hidden layer 3
                hidd_3 = tf.keras.layers.Dense(units=neuron_l3, activation='relu')(hidd_2)

                # Check the fourth layer
                if neuron_l4 == 0:
                    # Output layer
                    outputs = tf.keras.layers.Dense(units=1, activation='linear')(hidd_3)

                    # Define the network
                    model = tf.keras.Model(inputs=inputs, outputs=outputs)

                    # Return the model
                    return model

                else:
                    # Hidden layer 4
                    hidd_4 = tf.keras.layers.Dense(units=neuron_l4, activation='relu')(hidd_3)

                    # Output layer
                    outputs = tf.keras.layers.Dense(units=1, activation='linear')(hidd_4)

                    # Define the network
                    model = tf.keras.Model(inputs=inputs, outputs=outputs)

                    # Return the model
                    return model


# Function to create DEEP LSTM Recurrent Neural Networks
def create_deeplstmmodel(neuron_l1, neuron_l2, neuron_l3, window_size, n_features):
    """
        This functions creates a Deep LSTM Recurrent Neural Network

        Input:
        - neuron_l1 (int): number of neurons desire in the first layer, we check
                         if this number is zero in that case we show an error message
        - neuron_l2 (int): number of neurons desire in the second layer, if this number
                         is zero we don't add the second layer
        - neuron_l3 (int): number of neurons desire in the third layer, if this number
                         is zero we don't add the third layer
        - window_size (int): contains the dimension of the window
        - n_features (int): contains the number of features

        Output
        - model (keras.Model): The neural network
    """
    # Create the network
    # Input layer
    inputs = tf.keras.layers.Input(shape=(window_size, n_features))

    # Hidden layers
    if neuron_l1 == 0:
        print("Error - First Layer (l1) can't have 0 neurons")
    else:
        # Check the second layer
        if neuron_l2 == 0:
            # Hidden layer 1
            hidd_1 = tf.keras.layers.LSTM(units=neuron_l1, return_sequences=False)(inputs)

            # Output layer
            outputs = tf.keras.layers.Dense(units=1, activation='linear')(hidd_1)

            # Define the network
            model = tf.keras.Model(inputs=inputs, outputs=outputs)

            # Return the model
            return model

        else:
            # Hidden layer 1
            hidd_1 = tf.keras.layers.LSTM(units=neuron_l1, return_sequences=True)(inputs)

            # Check the third layer
            if neuron_l3 == 0:
                # Hidden layer 2
                hidd_2 = tf.keras.layers.LSTM(units=neuron_l2, return_sequences=False)(hidd_1)

                # Output layer
                outputs = tf.keras.layers.Dense(units=1, activation='linear')(hidd_2)

                # Define the network
                model = tf.keras.Model(inputs=inputs, outputs=outputs)

                # Return the model
                return model

            else:
                # Hidden layer 2
                hidd_2 = tf.keras.layers.LSTM(units=neuron_l2, return_sequences=True)(hidd_1)

                # Hidden layer 3
                hidd_3 = tf.keras.layers.LSTM(units=neuron_l3, return_sequences=False)(hidd_2)

                # Output layer
                outputs = tf.keras.layers.Dense(units=1, activation='linear')(hidd_3)

                # Define the network
                model = tf.keras.Model(inputs=inputs, outputs=outputs)

                # Return the model
                return model


# Function to create Deep GRU Models
def create_deepgrumodel(neuron_l1, neuron_l2, neuron_l3, neuron_l4, window_size, n_features):
    """
        This functions creates a Deep GRU Recurrente Neural Network

        Input:
        - neuron_l1 (int): number of neurons desire in the first layer, we check
                 if this number is zero in that case we show an error message
        - neuron_l2 (int): number of neurons desire in the second layer, if this number
                 is zero we don't add the second layer
        - neuron_l3 (int): number of neurons desire in the third layer, if this number
                 is zero we don't add the third layer
        - neuron_l4 (int): number of neurons desire in the fourth layer, if this number
                 is zero we don't add the fourth layer
        - window_size (int): contains the dimension of the window
        - n_features (int): contains the number of features

        Output
        - model (keras.Model): The neural network
    """
    # Create the network
    # Input layer
    inputs = tf.keras.layers.Input(shape=(window_size, n_features))

    # Hidden layers
    if neuron_l1 == 0:
        print("Error - First Layer (l1) can't have 0 neurons")
    else:
        # Check the second layer
        if neuron_l2 == 0:
            # Hidden layer 1
            hidd_1 = tf.keras.layers.GRU(units=neuron_l1, return_sequences=False)(inputs)

            # Output layer
            outputs = tf.keras.layers.Dense(units=1, activation='linear')(hidd_1)

            # Define the network
            model = tf.keras.Model(inputs=inputs, outputs=outputs)

            # Return the model
            return model

        else:
            # Hidden layer 1
            hidd_1 = tf.keras.layers.GRU(units=neuron_l1, return_sequences=True)(inputs)

            # Check the third layer
            if neuron_l3 == 0:
                # Hidden layer 2
                hidd_2 = tf.keras.layers.GRU(units=neuron_l2, return_sequences=False)(hidd_1)

                # Output layer
                outputs = tf.keras.layers.Dense(units=1, activation='linear')(hidd_2)

                # Define the network
                model = tf.keras.Model(inputs=inputs, outputs=outputs)

                # Return the model
                return model

            else:
                # Hidden layer 2
                hidd_2 = tf.keras.layers.GRU(units=neuron_l2, return_sequences=True)(hidd_1)

                # Check the fourth layer
                if neuron_l4 == 0:
                    # Hidden layer 3
                    hidd_3 = tf.keras.layers.GRU(units=neuron_l3, return_sequences=False)(hidd_2)

                    # Output layer
                    outputs = tf.keras.layers.Dense(units=1, activation='linear')(hidd_3)

                    # Define the network
                    model = tf.keras.Model(inputs=inputs, outputs=outputs)

                    # Return the model
                    return model

                else:
                    # Hidden layer 3
                    hidd_3 = tf.keras.layers.GRU(units=neuron_l3, return_sequences=True)(hidd_2)

                    # Hidden layer 4
                    hidd_4 = tf.keras.layers.GRU(units=neuron_l4, return_sequences=False)(hidd_3)

                    # Output layer
                    outputs = tf.keras.layers.Dense(units=1, activation='linear')(hidd_4)

                    # Define the network
                    model = tf.keras.Model(inputs=inputs, outputs=outputs)

                    # Return the model
                    return model


# Function DEEP Simple RNN model
def create_simplernnmodel(neuron_l1, neuron_l2, neuron_l3, neuron_l4, window_size, n_features):
    """
    This functions creates a Simple RNN Recurrent Neural Network

    Input:
    - neuron_l1 (int): number of neurons desire in the first layer, we check
                 if this number is zero in that case we show an error message
    - neuron_l2 (int): number of neurons desire in the second layer, if this number
                 is zero we don't add the second layer
    - neuron_l3 (int): number of neurons desire in the third layer, if this number
                 is zero we don't add the third layer
    - neuron_l4 (int): number of neurons desire in the fourth layer, if this number
                 is zero we don't add the fourth layer
    - window_size (int): contains the dimension of the window
    - n_features (int): contains the number of features

    Output
    - model (keras.Model): The neural network
    """
    # Create the network
    # Input layer
    inputs = tf.keras.layers.Input(shape=(window_size, n_features))

    # Hidden layers
    if neuron_l1 == 0:
        print("Error - First Layer (l1) can't have 0 neurons")
    else:
        # Check the second layer
        if neuron_l2 == 0:
            # Hidden layer 1
            hidd_1 = tf.keras.layers.SimpleRNN(units=neuron_l1, return_sequences=False)(inputs)

            # Output layer
            outputs = tf.keras.layers.Dense(units=1, activation='linear')(hidd_1)

            # Define the network
            model = tf.keras.Model(inputs=inputs, outputs=outputs)

            # Return the model
            return model

        else:
            # Hidden layer 1
            hidd_1 = tf.keras.layers.SimpleRNN(units=neuron_l1, return_sequences=True)(inputs)

            # Check the third layer
            if neuron_l3 == 0:
                # Hidden layer 2
                hidd_2 = tf.keras.layers.SimpleRNN(units=neuron_l2, return_sequences=False)(hidd_1)

                # Output layer
                outputs = tf.keras.layers.Dense(units=1, activation='linear')(hidd_2)

                # Define the network
                model = tf.keras.Model(inputs=inputs, outputs=outputs)

                # Return the model
                return model

            else:
                # Hidden layer 2
                hidd_2 = tf.keras.layers.SimpleRNN(units=neuron_l2, return_sequences=True)(hidd_1)

                # Check the fourth layer
                if neuron_l4 == 0:
                    # Hidden layer 3
                    hidd_3 = tf.keras.layers.SimpleRNN(units=neuron_l3, return_sequences=False)(hidd_2)

                    # Output layer
                    outputs = tf.keras.layers.Dense(units=1, activation='linear')(hidd_3)

                    # Define the network
                    model = tf.keras.Model(inputs=inputs, outputs=outputs)

                    # Return the model
                    return model

                else:
                    # Hidden layer 3
                    hidd_3 = tf.keras.layers.SimpleRNN(units=neuron_l3, return_sequences=True)(hidd_2)

                    # Hidden layer 4
                    hidd_4 = tf.keras.layers.SimpleRNN(units=neuron_l4, return_sequences=False)(hidd_3)

                    # Output layer
                    outputs = tf.keras.layers.Dense(units=1, activation='linear')(hidd_4)

                    # Define the network
                    model = tf.keras.Model(inputs=inputs, outputs=outputs)

                    # Return the model
                    return model


# Function to create Deep Convolutional models
def create_deepconvmodel(filters_l1, filters_l2, filters_l3, window_size, n_features):
    """
        This functions creates a Convolutional Neural Network

        Input:
        - filters_l1 (int): number of filters desire in the first layer, we check
                     if this number is zero in that case we show an error message
        - filters_l2 (int): number of filters desire in the second layer, if this number
                     is zero we don't add the second layer
        - filters_l3 (int): number of filters desire in the third layer, if this number
                     is zero we don't add the third layer
        - window_size (int): contains the dimension of the window
        - n_features (int): contains the number of features

        Output
        - model (keras.Model): The neural network
    """
    # Create the network
    # Input layer
    inputs = tf.keras.layers.Input(shape=(window_size, n_features))

    # Hidden layers
    if filters_l1 == 0:
        print("Error - First Layer (l1) can't have 0 neurons")
    else:
        # Hidden layer 1
        hidd_1 = keras.layers.Conv1D(filters=filters_l1, kernel_size=3, activation='relu')(inputs)

        # Check the second layer
        if filters_l2 == 0:
            # Flatten layer
            flatten = keras.layers.Flatten()(hidd_1)

            # Hidden layers - Deep
            d_1 = tf.keras.layers.Dense(128, activation='relu', name='dense1')(flatten)
            d_2 = tf.keras.layers.Dense(64, activation='relu', name='dense2')(d_1)

            # Output layer
            outputs = tf.keras.layers.Dense(units=1, activation='linear')(d_2)

            # Define the network
            model = tf.keras.Model(inputs=inputs, outputs=outputs)

            # Return the model
            return model

        else:
            # Hidden layer 2
            hidd_2 = keras.layers.Conv1D(filters=filters_l2, kernel_size=3, activation='relu')(hidd_1)

            # Check the third layer
            if filters_l3 == 0:
                # Flatten layer
                flatten = keras.layers.Flatten()(hidd_2)

                # Hidden layers - Deep
                d_1 = tf.keras.layers.Dense(128, activation='relu', name='dense1')(flatten)
                d_2 = tf.keras.layers.Dense(64, activation='relu', name='dense2')(d_1)

                # Output layer
                outputs = tf.keras.layers.Dense(units=1, activation='linear')(d_2)

                # Define the network
                model = tf.keras.Model(inputs=inputs, outputs=outputs)

                # Return the model
                return model

            else:
                # Hidden layer 3
                hidd_3 = keras.layers.Conv1D(filters=filters_l3, kernel_size=3, activation='relu')(hidd_2)

                # Flatten layer
                flatten = keras.layers.Flatten()(hidd_3)

                # Hidden layers - Deep
                d_1 = tf.keras.layers.Dense(128, activation='relu', name='dense1')(flatten)
                d_2 = tf.keras.layers.Dense(64, activation='relu', name='dense2')(d_1)

                # Output layer
                outputs = tf.keras.layers.Dense(units=1, activation='linear')(d_2)

                # Define the network
                model = tf.keras.Model(inputs=inputs, outputs=outputs)

                # Return the model
                return model
