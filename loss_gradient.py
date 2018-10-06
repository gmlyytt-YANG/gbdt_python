# -*- coding:utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Yang Li. All Rights Reserved
#
########################################################################

"""
File: loss_gradient.py
Author: Yang Li
Date: 2018/10/05 21:32:31
Descriptor:
    loss and gradient class definition.
"""

import numpy as np
import abc


class LossBase(metaclass=abc.ABCMeta):
    """Abstract class of loss."""
    @abc.abstractmethod
    def loss(self, y_i, f_x):
        """Compute loss of y and f_x.
        :param y_i:
        :param f_x:
        :return:
        """
        pass

    @abc.abstractmethod
    def gradient(self, y_i, f_x):
        """Gradient compute
        :param y_i:
        :param f_x:
        :return:
        """
        pass


class SquareLoss(LossBase):
    """Square Loss definition."""
    def loss(self, y_i, f_x):
        """Compute square loss of y and f_x.
        :param y_i:
        :param f_x:
        :return:
        """
        length = len(y_i)
        return float(1) / float(length) * np.sum((y_i - f_x) ** 2)

    def gradient(self, y_i, f_x):
        """Gradient compute of 1/2 (y_i - f_x)^2
        :param y_i:
        :param f_x:
        :return:
        """
        return f_x - y_i
