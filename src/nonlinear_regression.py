""" Non-linear Model Regression """

# - import modules - #
import utils
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers


if __name__ == '__main__':
    """ 
        Store arrays from data.npz in x and y.
        Split data into train and test set.
        Distinguish between features and labels.
    """
    X, y = utils.load_data("../data/data.npz")

    train_set, test_set, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)
    train_set = pd.DataFrame(train_set, columns=['x_1', 'x_2'])
    test_set = pd.DataFrame(test_set, columns=['x_1', 'x_2'])

    """
        Non-linear model.
        It is used a Sequential model with three densely connected hidden layers, 
        and an output layer that returns a single, continuous value. 
        The model building steps are wrapped in a function, build_model.
        
        In the model is implemented the early stopping.
    """
    def build_model(neurons=64, activation='relu'):
        model = keras.Sequential([
            layers.Dense(neurons, activation=activation, input_shape=[len(train_set.keys())]), # 1st hidden layer
            layers.Dense(neurons, activation=activation),                                      # 2nd hidden layer
            layers.Dense(neurons, activation=activation),                                      # 3rd hidden layer
            layers.Dense(1)                                                                    # output layer
        ])

        optimizer = tf.keras.optimizers.RMSprop(0.0001)  # optimizer and learning rate

        model.compile(loss='mse',                        # defined loss function
                      optimizer=optimizer,
                      metrics=['mse'])                   # metrics monitored during training

        """ 
           Normalize data.
           Normalized data will be used to train the model.
           
            Note: these statistics are intentionally generated from only the training data set.
        """

        train_stats = train_set.describe().T
        def norm(x):
            return (x - train_stats['mean']) / train_stats['std']
        normed_train_set = np.array(norm(train_set))

        return model

    model = build_model()
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, mode='auto')
    model.fit(x=train_set, y=train_labels,
              epochs=1000, verbose=0, callbacks=[early_stopping],
              validation_data=(test_set, test_labels))


    """ 
        Save the model in deliverable.
    """
    utils.save_keras_model(model, '../deliverable/nonlinear_model.pickle')





