# -*- coding:utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Yang Li. All Rights Reserved
#
########################################################################

"""
File: gbdt.py
Author: Yang Li
Date: 2018/10/02 21:47:31
Descriptor:
    gbdt self-realization.
"""

import numpy as np
import sys
from sklearn.datasets import load_boston
import warnings

import criterion
from decision_tree import CartRregressor
from base_regressor import LinearRegressor
import loss_gradient
from plot import AnimationPlot, ImagePlot
import util

warnings.simplefilter("error")


class GradientBosstingRegressor(object):
    """GBDT class define.

    This version of gbdt can achieve subsample,
    which is better to deal with over-fitting.

    Attributes:
        sub_sample: subsample of data-set.
        n_estimators: tree num.
        learning_rate: lr of sgd.
        loss_gradient: loss gradient parameters.
        regressors: n_estimators regressor(which could be any base regressor) vector.
    """

    def __init__(self, params):
        """Constructor of class.
        :param params:
        """
        self.sub_sample = params['sub_sample']
        self.n_estimators = params['n_estimators']
        self.learning_rate = params['learning_rate']
        self.loss_gradient = params['loss_gradient']
        # this version of gbdt could use any regressor, not just decision tree.
        self.regressors = \
            [params['regressor'](params['regressor_param'])] * self.n_estimators
        self.__check_params()

    def __check_params(self):
        """Check parameters' legality.
        :return:
        """
        self.sub_sample = np.clip(self.sub_sample, 0, 1)
        self.learning_rate = np.clip(self.learning_rate, 0, 1)
        if self.n_estimators < 1:
            self.n_estimators = 1

    def __get_dataset(self, dataset):
        """Get data by sampling without replacement
        :param dataset:
        :return: subsampled dataset.
        """
        np.random.shuffle(dataset)
        dataset_sample = dataset[:int(self.sub_sample * len(dataset))]
        return dataset_sample

    def fit(self, dataset, gui="cmd"):
        """Fit core function.
        :param dataset:
        :return: loss_list: training process
        """
        dataset = dataset.astype(float)
        if len(dataset) == 0:
            return
        loss_list = []
        loss_max = sys.float_info.min
        for i in range(self.n_estimators):
            dataset_sample = self.__get_dataset(dataset)
            annotations = dataset_sample[:, -1].copy()
            regressor = self.regressors[i]
            if i == 0:
                regressor.fit(dataset_sample)
                labels = [regressor.predict(data[:-1]) for data in dataset_sample]
            else:
                last_regressor = self.regressors[i - 1]
                last_regressor.fit(dataset_sample)
                # f_{m-1}(x) when m > 1
                last_labels = [last_regressor.predict(data[:-1]) for data in dataset_sample]
                # r_{mi}
                targets = [-self.loss_gradient.gradient(dataset_sample[i, -1], last_labels[i]) \
                           for i in range(len(last_labels))]
                # update targets
                util.update_last_column(dataset_sample, targets)
                regressor.fit(dataset_sample)
                new_labels = [regressor.predict(data[:-1]) for data in dataset_sample]
                # f_{m}(x) = f_{m-1}(x) + r_{mi}
                labels = np.array(last_labels) + np.array(new_labels)
                util.update_last_column(dataset_sample, labels)
            loss = self.loss_gradient.loss(annotations, labels)
            loss_max = max(loss, loss_max)
            if gui == "cmd":
                print('iteration={}, loss={}'.format(i, loss))
            loss_list.append(loss)
        if gui == "image":
            ImagePlot().plot(loss_list, 0, self.n_estimators, 0, loss_max)
        # just for fun
        elif gui == "animation":
            AnimationPlot().plot(loss_list, 0, self.n_estimators, 0, loss_max)

        return loss_list


if __name__ == "__main__":
    matrix = load_boston()
    data = matrix['data']
    target = matrix['target']
    matrix = np.hstack((data, np.expand_dims(target, 1)))
    params_tree = {
        'sub_sample': 0.7,
        'n_estimators': 100,
        'learning_rate': 0.00001,
        'regressor': CartRregressor,
        'regressor_param': {
            'max_depth': 4,
            'criterion': criterion.VarianceReduce()
        },
        'loss_gradient': loss_gradient.SquareLoss()
    }
    params_lr = {
        'sub_sample': 0.7,
        'n_estimators': 100,
        'learning_rate': 0.00001,
        'regressor': LinearRegressor,
        'regressor_param': None,
        'loss_gradient': loss_gradient.SquareLoss()
    }
    gbdt_regressor = GradientBosstingRegressor(params_tree)
    loss_list_tree = gbdt_regressor.fit(matrix, gui="image")
    gbdt_regressor = GradientBosstingRegressor(params_lr)
    loss_list_lr = gbdt_regressor.fit(matrix, gui="image")
