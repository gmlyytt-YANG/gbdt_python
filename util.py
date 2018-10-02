# -*- coding:utf-8 -*-

import numpy as np


def float_judge(a, b):
    eps = 1e-8
    if -eps <= a - b <= eps:
        return True
    return False


def matrix_same(matrix):
    first = matrix[0]
    for elem in matrix:
        if not float_judge(elem, first):
            return False
    return True


def square_loss_gradient(y_i, f_x):
    return f_x - y_i


def square_loss_compute(y, f_x):
    length = len(y)
    return float(1) / float(length) * np.sum((y - f_x) ** 2)


def update_last_column(matrix, new_col_data):
    if len(matrix) != len(new_col_data):
        return
    for i in range(len(matrix)):
        matrix[i, -1] = new_col_data[i]
