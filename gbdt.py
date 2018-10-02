# -*- coding:utf-8 -*-

import numpy as np
import warnings
from decision_tree import DecisionTree
import util
from sklearn.datasets import load_boston

warnings.simplefilter("error")


class GBDT(object):
    def __init__(self, params):
        self.sub_sample = params['sub_sample']
        self.n_estimators = params['n_estimators']
        self.learning_rate = params['learning_rate']
        self.max_depth = params['max_depth']
        self.mvr = params['mvr']
        self.regressors = [DecisionTree(params['max_depth'], params['mvr']) \
                           for i in range(self.n_estimators)]
        self.__check_params()

    def __check_params(self):
        self.sub_sample = np.clip(self.sub_sample, 0, 1)
        self.learning_rate = np.clip(self.learning_rate, 0, 1)
        if self.n_estimators < 1:
            self.n_estimators = 1

    def __get_dataset(self, dataset):
        np.random.shuffle(dataset)
        dataset_sample = dataset[:int(self.sub_sample * len(dataset))]
        return dataset_sample

    def fit(self, dataset):
        dataset = dataset.astype(float)
        if len(dataset) == 0:
            return
        labels = []
        for i in range(self.n_estimators):
            dataset_sample = self.__get_dataset(dataset)
            annotations = dataset_sample[:, -1].copy()
            regressor = self.regressors[i]
            if i == 0:
                regressor.fit(dataset_sample)
                labels = [regressor.predict(data[:-1]) for data in dataset_sample]
            else:
                last_result = labels.copy()  # f_{m-1}(x) when m > 1
                targets = [-util.square_loss_gradient(dataset_sample[i, -1], labels[i]) \
                           for i in range(len(labels))]  # r_{mi}
                util.update_last_column(dataset_sample, targets)
                regressor.fit(dataset_sample)
                new_labels = [regressor.predict(data[:-1]) for data in dataset_sample]
                labels = np.array(last_result) + np.array(new_labels)  # f_{m}(x) = f_{m-1}(x) + r_{mi}
                util.update_last_column(dataset_sample, labels)
            loss = util.square_loss_compute(annotations, labels)
            print("iteration {}, loss = {}".format(i, loss))


if __name__ == "__main__":
    matrix = load_boston()
    data = matrix['data']
    target = matrix['target']
    matrix = np.hstack((data, np.expand_dims(target, 1)))
    params = {
        'sub_sample': 0.7,
        'n_estimators': 500,
        'learning_rate': 0.00001,
        'max_depth': 4,
        'mvr': 0.03,
    }
    gbdt_regressor = GBDT(params)
    gbdt_regressor.fit(matrix)
