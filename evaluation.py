import numpy as np
import matplotlib.pyplot as plt


def f1_score(y_true, y_pred):
    """
    calculate f1_score
    2*recall*precision / recall + precision
    :param y_true: array holding true labels
    :param y_pred: array holding predicted labels
    :return: returns f1_score of binary classification task with true labels y_true and predicted labels y_pred
    """
    recall, precision = recall_precision(y_true, y_pred)
    return 2*recall*precision / (recall+precision)


def recall_precision(y_true, y_pred):
    """
    calculate recall
    TP / TP + FN
    and precision
    TP / TP + FP
    :param y_true: array holding true labels
    :param y_pred: array holding predicted labels
    :return: returns recall of binary classification task
    """
    TN, FP, FN, TP = binary_confusion_matrix(y_true, y_pred)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    return recall, precision


def binary_confusion_matrix(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    return TN, FP, FN, TP


def rmse(y_true, y_pred):
    """

    :param y_true: array holding true labels
    :param y_pred: array holding predicted labels
    :return: returns RMSE of regression task with true labels y_true and predicted labels y_pred
    """
    return (sum((y_true - y_pred) ** 2) / len(y_true)) ** 0.5


def visualize_results(k_list, scores, metric_name, title, path):
    pass

