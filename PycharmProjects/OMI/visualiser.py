# Visualiser: Plot functions
# Xingchen Wan | Xingchen.Wan@st-annes.ox.ac.uk | Oxford-Man Institute of Quantitative Finance

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.colors as mcolors
import datetime
import math
import networkx as nx
from utilities import create_sub_obj
from collections import OrderedDict


def get_color_map(data_labels, categories):
    color_map_num2text = ['b', 'g', 'r', 'c', 'm', 'y', 'olive', 'coral', 'brown', 'gray', 'indigo']
    color_map ={ x:'k' for x in data_labels}
    assert isinstance(categories, dict)
    color_map_keys = list(set(categories.values()))
    color_map_values = [color_map_num2text[i] if i < len(color_map_num2text) else 'k' for i in
                        range(len(color_map_keys))]
    color_map_dict = dict(zip(color_map_keys, color_map_values))
    for name in data_labels:
        try:
            color_map[name] = color_map_dict[categories[name]]
        except KeyError:
            pass
    return color_map


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

    assert len(data) == len(data_labels)
    # Default colour scheme

    c = ['k']*len(data_labels)
    if categories:
        color_map = get_color_map(data_labels, categories)
        c = [color_map[name] for name in data_labels]

    axis = np.linspace(0, 100, len(data))
    fig, ax = plt.subplots()
    ax.scatter(axis, data, c=c, s=10)
    plt.axhline(0, color="gray")
    plt.axhline(np.nanmean(data), color="red")
    plt.axhline(np.nanmedian(data), color="blue")
    if x_label: plt.xlabel(x_label)
    if y_label: plt.ylabel(y_label)
    if categories:
        labels = list(categories.values())

    for i in range(len(data)):
        if categories:
            ax.scatter(axis[i], data[i], c=c[i], label=labels[i] if labels[i] not in
                                                                        plt.gca().get_legend_handles_labels()[1] else '')
        else:
            ax.scatter(axis[i], data[i], c=c[i])
        ax.annotate(data_labels.iloc[i], (axis[i], data[i]), fontsize=6)
    plt.legend(prop={'size': 6})


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


def plot_network(full_data_obj, names=[], start_date=None, end_date=None, categories=None, threshold=90,
                 group_by=None):

    if not group_by:
        group_by = 'category'
    sub_obj = create_sub_obj(full_data_obj, start_date=start_date, end_date=end_date)
    G = sub_obj.build_occurrence_network_graph(focus_iterable=names)
    color_map = get_color_map(names, categories)
    c = [color_map[name] for name in G.nodes()] if categories else ['k']*len(names)
    pos = nx.spring_layout(G)
    edgelist = G.edges(data=True)
    if group_by == 'centrality':
        centrality = list(nx.katz_centrality(G, weight='weight').values())
        vmin = np.min(centrality)
        vmax = np.max(centrality)

    if threshold:
        if isinstance(threshold, int):
            weight_list = np.array([x[2]['weight'] for x in edgelist])
            threshold = np.percentile(weight_list, threshold)
            edgelist = [x for x in edgelist if x[2]['weight'] >= threshold]
        else:
            edgelist = [x for x in edgelist if x[2]['weight'] >= threshold]
    if group_by == 'category':
        nx.draw(G, pos, edgelist=edgelist, node_color=c, with_labels=True)
    elif group_by == 'centrality':
        nx.draw(G, pos, edgelist=edgelist,
                cmap=plt.plasma(),
                node_color=centrality,
                with_labels=True)
        sm = plt.cm.ScalarMappable(cmap=plt.plasma(), norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        plt.colorbar(sm)
    else:
        raise NotImplemented()


def plot_distribution_around_events(df):
    import matplotlib.mlab as mlab

    def gen_caption(days_to_event):
        if days_to_event < 0:
            return str(abs(days_to_event))+'d before event'
        elif days_to_event == 0:
            return "Event day"
        else:
            return str(days_to_event)+'d after event'

    numeric_col_names = [i for i in list(df.columns) if isinstance(i, int)]
    subplot_width = 2
    subplot_height = math.ceil(len(numeric_col_names) / subplot_width)
    df = df[numeric_col_names]
    i = 0
    for col in numeric_col_names:
        ax = plt.subplot(subplot_width, subplot_height, i+1)
        data = df[col].values
        data = data[~np.isnan(data)]
        mu = np.mean(data)
        sigma = np.std(data)
        n, bins, patches = plt.hist(data, 50, normed=1)
        y = mlab.normpdf(bins, mu, sigma)
        l = plt.plot(bins, y, 'r--', linewidth=1)
        plt.xlabel('CAR')
        plt.axvline(mu, color='red')
        ax.set_title(gen_caption(col))
        i += 1


def plot_events(*event_summaries, descps, test_chosen='t-test', mode='rtn'):

    def plot_event(event_summary, descp, test_chosen, mode):
        if mode == 'vol':
            event_summary = event_summary[[i for i in list(event_summary.columns) if isinstance(i, int)]]
        xi = [x+1 for x in list(event_summary.columns)]
        mean_val = event_summary.loc['mean', :].values
        plt.plot(xi, mean_val, label=descp)
        test_stats = event_summary.loc[test_chosen, :].values
        sig_idx = np.argwhere((test_stats < 0.05) & (test_stats > 0.01))
        v_sig_idx = np.argwhere(test_stats <= 0.01)
        if len(sig_idx):
            plt.plot(sig_idx+xi[0], mean_val[[sig_idx]], 'o', color='orange', label='p < 0.05')
        if len(v_sig_idx):
            plt.plot(v_sig_idx+xi[0], mean_val[[v_sig_idx]], 'o', color='red', label='p < 0.01')

    if len(descps) != len(event_summaries):
        raise ValueError("Descp length must match number of event summaries passed!")
    for i in range(len(event_summaries)):
        plot_event(event_summaries[i], descps[i], test_chosen=test_chosen, mode=mode)
    if mode == 'rtn':
        plt.axhline(0, color='k', linestyle='dashed')
        plt.axvline(0, color='k', linestyle='dashed')
        plt.xlabel('Days to the event')
        plt.ylabel('Cumulative Abnormal Return (CAR)')
    elif mode == 'vol':
        plt.axhline(0, color='k', linestyle='dashed')
        plt.axvline(0, color='k', linestyle='dashed')
        plt.xlabel('Days to the event')
        plt.ylabel('Volatility')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())


def plot_x_events(event_summaries, test_chosen='t-test'):
    for name, summary in event_summaries.items():
        plot_events(summary, descps=[name], test_chosen=test_chosen, mode='rtn')