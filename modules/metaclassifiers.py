from sklearn import base

### UNWEIGHTED AVERAGE

class UnweightedAverage(base.BaseEstimator, base.TransformerMixin):

    '''

    Metaclassifier - returns the linear unweighted average of multiple model predictions

    -----------

    - Accepts predictions in the form of a numpy array of shape (n_samples, n_models*n_classes)

    - Returns ensembled predictions in the form of a numpy array of shape (n_samples, n_classes)

    -----------

    Methods:

    __init__(self, n_classes = 9)

    fit(self, X=None, y=None)

    predict(self, X)


    Arguments:

    n_classes = number of classes
                Note: dimension 1 of input array must be divisible by n_classes

    X = input predictions, numpy array of shape (n_samples, n_models*n_classes)

    '''

    def __init__(self, n_classes=9):
        self.n_classes = n_classes

    def fit(self, X=None, y=None):
        return self

    def predict_proba(self, X):

        import numpy as np

        num_models = X.shape[1] / self.n_classes

        predictions = np.sum(np.split(X, int(num_models), axis=1), axis=0) / num_models

        return predictions