import sys

import cross_validation
import data
import evaluation
import knn
import numpy as np


def main(argv):
    df = data.load_data(argv[1])
    folds = data.get_folds()
    # part A - Classification
    print("Part1 - Classification")
    X = (data.add_noise(df[["t1", "t2", "wind_speed", "hum"]])).to_numpy()
    y = data.adjust_labels(list(df["season"]))
    k_list = [3, 5, 11, 25, 51, 75, 101]
    metric = evaluation.f1_score
    model = knn.ClassificationKNN
    means, sds = cross_validation.model_selection_cross_validation(model, k_list, X, y, folds, metric)
    for mean, sd, k in zip(means, sds, k_list):
        print(f"k={k}, mean score: {mean:.4f}, std of scores: {sd:.4f}")
    evaluation.visualize_results(k_list, means, "f1_score", "Classification", "./Classification")
    print()
    # part B - Regression
    print("Part2 - Regression")
    folds = data.get_folds()
    X = (data.add_noise(df[["t1", "t2", "wind_speed"]])).to_numpy()
    y = list(df["hum"])
    metric = evaluation.rmse
    model = knn.RegressionKNN
    means, sds = cross_validation.model_selection_cross_validation(model, k_list, X, y, folds, metric)
    for mean, sd, k in zip(means, sds, k_list):
        print(f"k={k}, mean score: {mean:.4f}, std of scores: {sd:.4f}")
    evaluation.visualize_results(k_list, means, "RMSE", "Regression", "./Regression")


if __name__ == '__main__':
    main(sys.argv)
