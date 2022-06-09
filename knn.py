import numpy as np
from scipy.stats import mode
from abc import abstractmethod
from data import StandardScaler as SS


class KNN:
    def __init__(self, k):
        """
        constructor
        :param k: k used for knn model
        """
        self.X_train = None
        self.y_train = None
        self.k = k
        self.ss = SS()  # initialize standard scaler object for model

    def fit(self, X_train, y_train):
        """
        calc mean and std
        :param X_train: calc stats for this param
        :param y_train: and this one too
        :return:
        """
        self.X_train = X_train
        self.y_train = y_train
        self.ss.fit(X_train)

    @abstractmethod
    def predict(self, X_test):
        pass

    def neighbours_indices(self, x):
        """
        find indices of k closest neighbors to x
        :param x: the point
        :return: np array of said indices
        """
        distances = list()
        X_train = self.ss.transform(self.X_train)  # lets transform
        for sample in X_train:  # get distances of all point from x
            dist = KNN.dist(x, sample)
            distances.append(dist)
        idx = np.argsort(np.array(distances))  # get only the first k
        return idx[:self.k]

    @staticmethod
    def dist(x1, x2):
        """
        euclidian distance method
        :param x1: np array
        :param x2: np array
        :return: euclidian distance between 2 vectors x1 and x2
        """
        return np.linalg.norm(x1 - x2)


class ClassificationKNN(KNN):
    def __init__(self, k):
        super().__init__(k)

    def predict(self, X_test):
        pred = list()
        X_test = self.ss.transform(X_test)
        for x in X_test:
            indices = self.neighbours_indices(x)  # index's of closest neighbors
            closest_labels = [self.y_train[index] for index in indices]  # labels of closest neighbors
            predicted_label = mode(closest_labels).mode[0]
            pred.append(predicted_label)  # get the mode and append to the list
        return pred


class RegressionKNN(KNN):
    def __init__(self, k):
        super().__init__(k)

    def predict(self, X_test):
        pred = list()
        X_test = self.ss.transform(X_test)
        for x in X_test:
            indices = self.neighbours_indices(x)
            sub_list = np.array([self.y_train[idx] for idx in indices])
            pred.append(sub_list.mean())
        return pred
