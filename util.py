# -*- coding:utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Yang Li. All Rights Reserved
#
########################################################################

"""
File: util.py
Author: Yang Li
Date: 2018/10/02 22:04:31
"""

import numpy as np


def float_judge(a, b):
    """Judge whether two float equaling.
    :param a:
    :param b:
    :return:
    """
    eps = 1e-8
    if -eps <= a - b <= eps:
        return True
    return False


def matrix_same(matrix):
    """Judge whether two float matrices equaling.
    :param matrix:
    :return:
    """
    first = matrix[0]
    for elem in matrix:
        if not float_judge(elem, first):
            return False
    return True


def square_loss_gradient(y_i, f_x):
    """Gradient of 1/2(x)^2
    :param y_i:
    :param f_x:
    :return:
    """
    return f_x - y_i


def square_loss_compute(y, f_x):
    """Compute square loss of y and f_x.
    :param y:
    :param f_x:
    :return:
    """
    length = len(y)
    return float(1) / float(length) * np.sum((y - f_x) ** 2)


def update_last_column(matrix, new_col_data):
    """Replace the last column of matrix with new_col_data.
    :param matrix:
    :param new_col_data:
    :return:
    """
    if len(matrix) != len(new_col_data):
        return
    for i in range(len(matrix)):
        matrix[i, -1] = new_col_data[i]
