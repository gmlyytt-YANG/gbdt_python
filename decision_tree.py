# -*- coding:utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Yang Li. All Rights Reserved
#
########################################################################

"""
File: decision_tree.py
Author: Yang Li
Date: 2018/10/02 21:57:31
"""

import numpy as np

from base_regressor import DecisionTree
import util


class CartRregressor(DecisionTree):
    """Cart Decision Tree class define"""

    def __init__(self, params):
        """Constructor."""
        super(CartRregressor, self).__init__(params)

    def create_tree(self, depth, data):
        """Create tree.
        :param depth:
        :param data:
        :return: tree_info: model parameter
        """
        if depth == 1:
            return np.mean(data[:, -1])
        if util.matrix_same(data[:, -1]):
            return data[0, -1]
        matrix_sum_axis1 = np.sum(data[:, :-1], axis=1)
        if util.matrix_same(matrix_sum_axis1):
            return np.mean(data[:, -1])
        best_feature_index, feature_threshold \
            = self.criterion.choose_best_feature(data)

        tree_info = {}
        left_data = data[data[:, best_feature_index] <= feature_threshold]
        right_data = data[data[:, best_feature_index] > feature_threshold]
        if len(left_data) and len(right_data):
            tree_info['left'] = self.create_tree(depth - 1, left_data)
            tree_info['right'] = self.create_tree(depth - 1, right_data)
            tree_info['feature'] = best_feature_index
            tree_info['threshold'] = feature_threshold
            return tree_info
        else:
            return np.mean(data[:, -1])

    def predict_core(self, test_data, tree_info):
        """Core function of Predict.
        :param test_data:
        :param tree_info:
        :return:
        """
        if not isinstance(tree_info, dict):
            return tree_info
        if len(test_data) <= tree_info['feature']:
            print('invalid data format')
            return
        if test_data[tree_info['feature']] > tree_info['threshold']:
            return self.predict_core(test_data, tree_info['right'])
        else:
            return self.predict_core(test_data, tree_info['left'])
