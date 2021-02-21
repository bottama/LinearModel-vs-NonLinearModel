import joblib
from keras import models
import keras
import numpy as np

def save_sklearn_model(model, filename):
    """
    Saves a Scikit-learn model to disk.
    Example of usage:

    :param model: the model to save;
    :param filename: string, path to the file in which to store the model.
    :return: the model.
    """
    joblib.dump(model, filename)


def load_sklearn_model(filename):
    """
    Loads a Scikit-learn model saved with joblib.dump.

    :param filename: string, path to the file storing the model.
    :return: the model.
    """
    model = joblib.load(filename)

    return model


def save_keras_model(model, filename):
    """
    Saves a Keras model to disk.
    Example of usage:

    :param model: the model to save;
    :param filename: string, path to the file in which to store the model.
    :return: the model.
    """
    models.save_model(model, filename)


def load_keras_model(filename):
    """
    Loads a compiled Keras model saved with models.save_model.

    :param filename: string, path to the file storing the model.
    :return: the model.
    """
    model = models.load_model(filename)

    return model

def load_data(filename):
    """
    Loads the data from a saved .npz file.
    ### YOU CAN NOT EDIT THIS FUNCTION ###

    :param filename: string, path to the .npz file storing the data.
    :return: two numpy arrays:
        - x, a Numpy array of shape (n_samples, n_features) with the inputs;
        - y, a Numpy array of shape (n_samples, ) with the targets.
    """
    data = np.load(filename)
    x = data['x']
    y = data['y']

    return x, y
