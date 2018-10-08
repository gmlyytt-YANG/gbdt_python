# -*- coding:utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Yang Li. All Rights Reserved
#
########################################################################

"""
File: criterion.py
Author: Yang Li
Date: 2018/10/08 19:43:31
"""

import abc
import numpy as np
import sys
import util


class BaseCriterion(metaclass=abc.ABCMeta):
    """Abstract class of Criterion."""

    @abc.abstractmethod
    def choose_best_feature(self, data):
        pass


class VarianceReduce(BaseCriterion):
    def choose_best_feature(self, data):
        """Choose best feature
        :param data:
        :return: best_feature_index:
        :return: feature_threshold:
        :return: feature_threshold
        """
        index = 0
        min_variance = sys.float_info.max
        best_feature_index, feature_threshold = 0, 0.0
        while index < len(data[0]) - 1:
            feature = data[:, index]
            label = data[:, -1]
            var, threshold = self.__variance(feature, label)
            if var < min_variance:
                min_variance = var
                best_feature_index, feature_threshold = index, threshold
            index += 1
        return best_feature_index, feature_threshold

    def __variance(self, feature, label):
        """Choose the best threshold of a feature according to
                distribution of label.
            :param feature: vector of data.
            :param label:
            :return: min_var: min variance of the feature
            :return: best_threshold:
            """
        data = np.vstack((feature, label)).T
        data_sorted = np.array(sorted(data, key=lambda x: x[0]))
        delta = float(data_sorted[-1][0] - data_sorted[0][0]) / 10
        threshold = data_sorted[0][0]
        best_threshold = data_sorted[0][0]
        if util.equal_judge(delta, 0.0):
            return 0.0, data_sorted[0][0]
        min_var = sys.float_info.max
        while threshold <= data_sorted[-1][0]:
            left = data_sorted[data_sorted[:, 0] <= threshold]
            right = data_sorted[data_sorted[:, 0] > threshold]
            left_var = 0 if len(left) == 0 else np.var(left[:, -1]) * len(left)
            right_var = 0 if len(right) == 0 else np.var(right[:, -1]) * len(right)
            var = left_var + right_var
            if var < min_var:
                min_var = var
                best_threshold = threshold
            threshold += delta

        return min_var, best_threshold
