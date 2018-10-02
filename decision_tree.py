# -*- coding:utf-8 -*-

import sys
import numpy as np
import util

def variance_reduce(feature, label):
    data = np.vstack((feature, label)).T
    data_sorted = np.array(sorted(data, key=lambda x: x[0]))
    delta = float(data_sorted[-1][0] - data_sorted[0][0]) / 10
    threshold = data_sorted[0][0]
    best_threshold = data_sorted[0][0]
    if util.float_judge(delta, 0):
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


class DecisionTree(object):
    def __init__(self, depth, min_variance_reduce):
        self.depth = depth
        self.mvr = min_variance_reduce
        self.tree_info = {}

    def choose_best_feature(self, data):
        index = 0
        min_variance_reduce = sys.float_info.max
        best_feature_index, feature_threshold = 0, 0.0
        while index < len(data[0]) - 1:
            feature = data[:, index]
            label = data[:, -1]
            var_red, threshold = variance_reduce(feature, label)
            if var_red < min_variance_reduce:
                min_variance_reduce = var_red
                best_feature_index, feature_threshold = index, threshold
            index += 1
        return best_feature_index, feature_threshold, min_variance_reduce

    def create_tree(self, depth, data):
        if depth == 1:
            return np.mean(data[:, -1])
        if util.matrix_same(data[:, -1]):
            return data[0, -1]
        index = 0
        feature_all_same = True
        while index < len(data[0]) - 1:
            if not util.matrix_same(data[:, index]):
                feature_all_same = False
                break
            index += 1
        if feature_all_same:
            return np.mean(data[:, -1])
        best_feature_index, feature_threshold, max_variance_reduce \
            = self.choose_best_feature(data)
        if max_variance_reduce <= self.mvr:
            return np.mean(data[:, -1])

        tree_info = {}
        left_data = data[data[:, best_feature_index] <= feature_threshold]
        right_data = data[data[:, best_feature_index] > feature_threshold]
        if len(left_data) > 0 and len(right_data) > 0:
            tree_info['left'] = self.create_tree(depth - 1, left_data)
            tree_info['right'] = self.create_tree(depth - 1, right_data)
            tree_info['feature'] = best_feature_index
            tree_info['threshold'] = feature_threshold
            return tree_info
        else:
            return np.mean(data[:, -1])

    def fit(self, data):
        max_depth = self.depth
        self.tree_info = self.create_tree(max_depth, data)

    def predict_core(self, test_data, tree_info):
        if not isinstance(tree_info, dict):
            return tree_info
        if len(test_data) <= tree_info['feature']:
            print('invalid data format')
            return
        if test_data[tree_info['feature']] > tree_info['threshold']:
            return self.predict_core(test_data, tree_info['right'])
        else:
            return self.predict_core(test_data, tree_info['left'])

    def predict(self, test_data):
        return self.predict_core(test_data, self.tree_info)