import numpy as np
from scipy import stats
from abc import abstractmethod
from data import StandardScaler as SS


class KNN:
    def __init__(self, k):
        self.X_train = None
        self.y_train = None
        self.k = k
        self.ss = SS()

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.ss.fit(X_train)

    @abstractmethod
    def predict(self, X_test):
        pass

    def neighbours_indices(self, x):
        distances = list()
        for sample in self.X_train:
            distances.append(KNN.dist(sample, x))
        idx = np.argpartition(np.array(distances), self.k)
        return idx[:self.k]

    @staticmethod
    def dist(x1, x2):
        return np.norm(x1 - x2)

    class ClassificationKNN(KNN):
        def __init__(self, k):
            super().__init__(k)

        def predict(self, X_test):
            pred = list()
            for x in X_test:
                indices = self.neighbours_indices(x)  # index's of closest neighbors
                closest_labels = [self.y_train[index] for index in indices]  # labels of closest neighbors
                pred.append(stats.mode(closest_labels))  # get the mode and append to the list
            return pred

    class RegressionKNN(knn.KNN):
        def __init__(self, k):
            super().__init__(k)

        def predict(self, X_test):
            pred = list()
            for x in X_test:
                indices = self.neighbours_indices(x)
                sub_list = np.array([self.y_train[idx] for idx in indices])
                pred.append(sub_list.mean())
            return pred
