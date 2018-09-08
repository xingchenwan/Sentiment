# Visualiser: Plot functions
# Xingchen Wan | Xingchen.Wan@st-annes.ox.ac.uk | Oxford-Man Institute of Quantitative Finance

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import datetime
import math


def plot_scatter(data, data_labels, x_label=None, y_label=None, categories=None):
    """
    Plot the scatter diagram of the correlation of different single names
    :param data: input data. pandas dataframe or numpy ndarray
    :param data_labels: label of each data point - usually the ticker or name of the entities in data. Should be of
    the same length as the data
    :param x_label: Optional. x-axis label of the scatter plot.
    :param y_label: Optional. y-axis label of the scatter plot
    :param categories: Optional {}. Sectorial or dynamic categorisation of the entities. Must be of the same length
    as the number of entities in data.
    :return:
    """
    color_map_num2text = ['b', 'g', 'r', 'c', 'm', 'y', 'olive', 'coral', 'brown', 'gray', 'indigo']

    assert len(data) == len(data_labels)
    color_map = ['k']*len(data)
    # Default colour scheme

    if categories:
        assert isinstance(categories, dict)
        color_map_keys = list(set(categories.values()))
        color_map_values = [color_map_num2text[i] if i < len(color_map_num2text) else 'k' for i in range(len(color_map_keys))]
        color_map_dict = dict(zip(color_map_keys, color_map_values))
        i = 0
        for name in data_labels:
            try:
                color_map[i] = color_map_dict[categories[name]]
            except KeyError:
                pass
            i += 1
    axis = np.linspace(0, 100, len(data))
    fig, ax = plt.subplots()
    ax.scatter(axis, data, c=color_map, s=10)
    plt.axhline(0, color="gray")
    plt.axhline(np.nanmean(data), color="red")
    plt.axhline(np.nanmedian(data), color="blue")
    if x_label: plt.xlabel(x_label)
    if y_label: plt.ylabel(y_label)

    for i in range(len(data_labels)):
        ax.annotate(data_labels.iloc[i], (axis[i], data[i]), fontsize=5)


def plot_single_name(name, *args, arg_names=[], start_date=None, end_date=None):
    if arg_names and len(arg_names) != len(args):
        raise ValueError("Length mismatch between arg_names and number of series supplied.")
    start_date = start_date if start_date and isinstance(start_date, datetime.date) else datetime.datetime(1929, 1, 1)
    end_date = end_date if end_date and isinstance(end_date, datetime.date) else datetime.datetime(2100, 1, 1)

    fig = plt.figure()
    subplot_width = 2
    subplot_height = math.ceil(len(args) / 2)
    i = 0
    for arg in args:
        if not isinstance(arg, pd.DataFrame) and not isinstance(arg, pd.Series):
            raise TypeError("Input arguments need to be Pandas DataFrames")
        elif not isinstance(arg.index, pd.DatetimeIndex):
            raise TypeError("Input series index needs to be DatetimeIndex")

    for arg in args:
        if isinstance(arg, pd.Series):
            series = arg
        elif name in list(arg.columns):
            series = arg[name]
        else:
            print(name, " not found in argument of position", args.index(arg))
            continue
        series = series[series.index >= start_date]
        series = series[series.index < end_date]
        ax = plt.subplot(subplot_width, subplot_height, i+1)
        plt.plot(series.index, series)
        plt.axhline(np.nanmean(series), color="red")
        plt.axhline(np.nanmedian(series), color="blue")
        plt.axhline(0, color="gray")

        if arg_names:
            ax.set_title(arg_names[i])
        i += 1

    fig.suptitle(name)




