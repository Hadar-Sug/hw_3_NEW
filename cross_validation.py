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
    for row, k in enumerate(k_list):
        model_k = model(k)
        scores = np.array(cross_validation_score(model_k, X, y, folds, metric))
        scores_mean.append(np.mean(scores))
        scores_sd.append(np.std(scores, ddof=1))
    return scores_mean, scores_sd


'''
Part1 - Classification
k=3, mean score: 0.6334, std of scores: 0.0471
k=5, mean score: 0.6282, std of scores: 0.0678
k=11, mean score: 0.6629, std of scores: 0.0331
k=25, mean score: 0.6771, std of scores: 0.0594
k=51, mean score: 0.7048, std of scores: 0.0249
k=75, mean score: 0.7144, std of scores: 0.0239
k=101, mean score: 0.7012, std of scores: 0.0175

Part2 - Regression
k=3, mean score: 13.0823, std of scores: 0.7603
k=5, mean score: 12.5599, std of scores: 0.5461
k=11, mean score: 11.9578, std of scores: 0.4589
k=25, mean score: 11.7461, std of scores: 0.6550
k=51, mean score: 11.9539, std of scores: 0.6480
k=75, mean score: 12.0184, std of scores: 0.5885
k=101, mean score: 12.1023, std of scores: 0.5602

'''
