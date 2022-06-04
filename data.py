import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
np.random.seed(2)


def load_data(path):
    """
    loads file from path to pandas df
    :param path: location of file
    :return: pandas df
    """
    return pd.read_csv(path)


def adjust_labels(y):
    return list(map(lambda x: 0 if x <= 1 else 1, y))


class StandardScaler:

    def __init__(self):
        self.sd_array = None
        self.mu_array = None

    ''' fit scaler by learning mean and standard deviation per feature '''
    def fit(self, X):
        self.mu_array = np.mean(X, axis=0)
        self.sd_array = np.std(X, axis=0)

    def transform(self, X):
        return (X - self.mu_array) / self.sd_array

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def add_noise(data):
    """
    :param data: dataset as np.array of shape (n, d) with n observations and d features
    :return: data + noise, where noise~N(0,0.001^2)
    """
    noise = np.random.normal(loc=0, scale=0.001, size=data.shape)
    return data + noise


def get_folds():
    """
    :return: sklearn KFold object that defines a specific partition to folds
    """
    return KFold(n_splits=5, shuffle=True, random_state=34)
