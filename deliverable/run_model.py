""" Run models """

# - import modules - #
import joblib
import numpy as np
import tensorflow as tf
from scipy.stats import ttest_rel

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

def load_model(filename):
    """
    Loads a Scikit-learn model saved with joblib.dump.
    This is just an example, you can write your own function to load the model.
    Some examples can be found in src/utils.py.

    :param filename: string, path to the file storing the model.
    :return: the model.
    """
    model = joblib.load(filename)

    return model

def evaluate_predictions(y_true, y_pred):
    """
    Evaluates the mean squared error between the values in y_true and the values
    in y_pred.
    ### YOU CAN NOT EDIT THIS FUNCTION ###

    :param y_true: Numpy array, the true target values from the test set;
    :param y_pred: Numpy array, the values predicted by your model.
    :return: float, the the mean squared error between the two arrays.
    """
    assert y_true.shape == y_pred.shape
    return ((y_true - y_pred) ** 2).mean()

"""
    Define functions to evaluate predicted labels from models.
    Given an input x it returns the predicted output y for the selected model. 
"""
def linear_model(x):
    linear_model_path = 'linear_regression.pickle'
    linear_model = load_model(linear_model_path)
    x = np.column_stack((x, np.sin(x[:, 0]) * x[:, 1]))
    y_pred = linear_model.predict(x)
    return y_pred

def nonlinear_model(x):
    nonlinear_model_path = 'nonlinear_model.pickle'
    nonlinear_model = tf.keras.models.load_model(nonlinear_model_path)
    y_pred = nonlinear_model.predict(x).reshape((-1,))
    return y_pred


def load_model(filename):
    """
    Loads a Scikit-learn model saved with joblib.dump.
    This is just an example, you can write your own function to load the model.
    Some examples can be found in src/utils.py.

    :param filename: string, path to the file storing the model.
    :return: the model.
    """
    model = joblib.load(filename)

    return model


if __name__ == '__main__':
    # Load the data
    # This will be replaced with the test data when grading the assignment
    data_path = "../data/data.npz"
    x, y = load_data(data_path)

    ############################################################################
    # EDITABLE SECTION OF THE SCRIPT: if you need to edit the script, do it here
    ############################################################################

    # Load the trained model
    # Store predicted labels

    """
        Linear model
    """
    pred_labels_linear = linear_model(x)

    """
        Non-linear model
    """
    pred_labels_nonlinear = nonlinear_model(x)

    """
        t-test and p-value.
        In this section we want to know which model is statistically better.
    """

    def model_selection():
        e_linear = (y - pred_labels_linear) ** 2
        e_nonlinear = (y - pred_labels_nonlinear) ** 2

        tt, p_val = ttest_rel(e_linear, e_nonlinear)

        if tt > 1.96 or tt < -1.96:
            print('t-test: T={:.2f}, p-value={}'.format(tt, p_val))
            print('Since T is outside 95% confidence interval (-1.96, 1.96) the two models are differents.'
                  'According to p-value and t-test the non-linear model is statistically better.')
        else:
            print('t-test: T={:.2f}, p-value={}'.format(tt, p_val))
            print('Since T is inside 95% confidence interval (-1.96, 1.96) the two models are not differents.'
                  'According to p-value and t-test the non-linear model is not statistically better.')
    model_selection()

    ############################################################################
    # STOP EDITABLE SECTION: do not modify anything below this point.
    ############################################################################


    # Evaluate the prediction using MSE

    mse_linear = evaluate_predictions(y, pred_labels_linear)
    mse_nonlinear = evaluate_predictions(y, pred_labels_nonlinear)

    print('MSE linear model: {}'.format(mse_linear))
    print('MSE non-linear model: {}'.format(mse_nonlinear))




