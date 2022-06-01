import numpy as np


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
    for train_idices, validation_indices in folds.split(X):
        X_train, X_test = X[train_idices], X[validation_indices]
        y_train, y_test = y[train_idices], y[validation_indices]
        y_pred = model.predict(X_test)
        scores.append(metric(y_test, y_pred))
    return scores


def model_selection_cross_validation(model, k_list, X, y, folds, metric):
    scores_mean = list()
    scores_sd = list()
    N = len(k_list)
    total_scores = np.zeros(N, 5)
    for row, k in enumerate(k_list):
        model_k = model(k)
        scores = np.array(cross_validation_score(model_k, X, y, folds, metric))
        total_scores[row] = scores
    scores_mean = np.mean(total_scores, axis=0)  # mean by columns? - check axis
    scores_mean = np.std(total_scores, axis=0) * ((N / (N - 1)) ** 0.5)  # sd by columns?
    return scores_mean, scores_sd


