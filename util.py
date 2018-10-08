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


def equal_judge(a, b):
    """Judge whether two nums equaling.
    :param a:
    :param b:
    :return:
    """
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    else:
        eps = 1e-8
        if -eps <= a - b <= eps:
            return True
        return False


def matrix_same(matrix):
    """Judge whether two matrices equaling.
    :param matrix:
    :return:
    """
    first = matrix[0]
    for elem in matrix:
        if not equal_judge(elem, first):
            return False
    return True


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
