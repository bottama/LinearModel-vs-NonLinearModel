""" Linear Model Regression """

# - import modules - #
import utils
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


if __name__ == '__main__':
    """ 
        Store arrays from data.npz in x and y.
        Split data into train and test set.
        Distinguish between features and labels.
        
        Note: since ordinary least squares is invariant, there is no need for standardization.
    """
    x, y =  utils.load_data("../data/data.npz")
    X = np.column_stack((x, np.sin(x[:, 0]) * x[:, 1]))

    train_set, test_set, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

    """ 
        Linear model.
            From sklearn train the linear regression model on the train set.         
    """

    regr = LinearRegression()
    regr.fit(train_set, train_labels)

    """
        Save the model in deliverable.
    """
    utils.save_sklearn_model(regr, '../deliverable/linear_regression.pickle')


