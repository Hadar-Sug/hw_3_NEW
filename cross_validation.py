import numpy as np

import knn


def cross_validation_score(model, X, y, folds, metric):
    """
    run cross validation on X and y with specific model by given folds. evaluate by given metric
    :param folds:
    :param model: object of type model (some KNN)
    :param X: np matrix - rows are samples and columns are features
    :param y: labels for samples
    :param metric: evaluation metric method i.e. f1 or RMSE etc...
    :return: list with score for each split
    """
    scores = list()
    for train_indices, validation_indices in folds.split(X):
        X_train, X_test = X[train_indices, :], X[validation_indices, :]
        y_train, y_test = [y[index] for index in train_indices], [y[index] for index in validation_indices]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append(metric(y_test, y_pred))
    return scores


def model_selection_cross_validation(model, k_list, X, y, folds, metric):
    scores_mean = list()
    scores_sd = list()
    N = len(k_list)
    for row, k in enumerate(k_list):
        model_k = model(k)
        scores = np.array(cross_validation_score(model_k, X, y, folds, metric))
        scores_mean.append(np.mean(scores))
        scores_sd.append(np.std(scores) * ((N / (N - 1)) ** 0.5))
    return scores_mean, scores_sd
