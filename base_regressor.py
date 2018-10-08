# -*- coding:utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Yang Li. All Rights Reserved
#
########################################################################

"""
File: base_regressor.py
Author: Yang Li
Date: 2018/10/08 18:06:31
Description: base class of decision tree
"""

import abc
import numpy as np
from sklearn.linear_model import LinearRegression


class DecisionTree(metaclass=abc.ABCMeta):
    """Decision Tree class define

    Atttibutes:
    depth: max depth of the tree.
    criterion: feature selection and node split criterion.
    tree_info: model parameter
    """

    def __init__(self, params):
        """Constructor.
        :param params:
        """
        self.depth = params['max_depth']
        self.criterion = params['criterion']
        self.tree_info = {}

    @abc.abstractmethod
    def create_tree(self, depth, data):
        """Create tree.
        :param depth:
        :param data:
        :return: tree_info: model parameter
        """
        pass

    def fit(self, data):
        """Fit the model
        :param data:
        :return:
        """
        max_depth = self.depth
        self.tree_info = self.create_tree(max_depth, data)

    @abc.abstractmethod
    def predict_core(self, test_data, tree_info):
        """Core function of Predict.
        :param test_data:
        :param tree_info:
        :return:
        """
        pass

    def predict(self, test_data):
        """Predict according to the test_data
        :param test_data:
        :return:
        """
        return self.predict_core(test_data, self.tree_info)


class LinearRegressor(object):
    """LinearRegressor class"""
    def __init__(self, params=None):
        """Constructor"""
        self.regressor = LinearRegression()

    def fit(self, data):
        """fit function"""
        X = data[:, :-1]
        y = data[:, -1]
        self.regressor.fit(X, y)

    def predict(self, test_data):
        """predict function"""
        # LinearRegression format
        if isinstance(test_data, list):
            test_data = np.array(test_data)
        return self.regressor.predict([test_data])[0]
