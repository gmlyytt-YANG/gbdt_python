# -*- coding:utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Yang Li. All Rights Reserved
#
########################################################################

"""
File: plot.py
Author: Yang Li
Date: 2018/10/09 17:57:00
Descriptor:
    plot function.
"""

import abc
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class BasePlot(metaclass=abc.ABCMeta):
    def data_gen(self, data_list):
        """Data generator.
        :param data_list:
        :return:
        """
        # if only one row, add index of it
        if isinstance(data_list[0], float) or isinstance(data_list[0], int):
            for index, elem in enumerate(data_list):
                yield index, elem
        else:
            for elem in data_list:
                yield elem[0], elem[1]

    @abc.abstractmethod
    def plot(self, data_list, x_min, x_max, y_min, y_max):
        """Plot function.
        :param data_list:
        :param x_min:
        :param x_max:
        :param y_min:
        :param y_max:
        :return:
        """
        pass


class ImagePlot(BasePlot):
    def plot(self, data_list, x_min, x_max, y_min, y_max):
        fig, ax = plt.subplots()
        ax.plot([i[1] for i in self.data_gen(data_list)], lw=2)
        ax.grid()
        plt.show()


class AnimationPlot(object):
    def plot(self, data_list, x_min, x_max, y_min, y_max):
        fig, ax = plt.subplots()
        line, = ax.plot([], [], lw=2)
        ax.grid()
        xdata, ydata = [], []

        def init():
            """Animation initialization."""
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(x_min, x_max)
            del xdata[:]
            del ydata[:]
            line.set_data(xdata, ydata)
            return line,

        def run(data):
            """Animation Runner."""
            # update the data
            t, y = data
            xdata.append(t)
            ydata.append(y)
            xmin, xmax = ax.get_xlim()
            if t >= xmax:
                ax.set_xlim(xmin, 2 * xmax)
                ax.figure.canvas.draw()
            line.set_data(xdata, ydata)
            return line,

        animation.FuncAnimation(fig, run, self.data_gen(data_list), blit=False, interval=10,
                                repeat=False, init_func=init)
        plt.show()
