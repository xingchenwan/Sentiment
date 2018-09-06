# Visualiser: Plot functions
import numpy as np
import matplotlib.pylab as plt


def plot_scatter(data, data_labels, x_label=None, y_label=None, categories=None):

    assert len(data) == len(data_labels)
    if categories:
        assert isinstance(categories, dict)

        color_map_keys = list(set(categories.values()))
        color_map_values = [i for i in range(len(color_map_keys))]
        color_map_dict = dict(zip(color_map_keys, color_map_values))
        color_map = np.array([color_map_dict[categories[name]] for name in data_labels])
    else:
        color_map = np.ones((len(data), 1))

    axis = np.linspace(0, 100, len(data))
    fig, ax = plt.subplots()
    ax.scatter(axis, data, c=color_map, s=10)
    plt.axhline(0, color="gray")
    plt.axhline(np.nanmean(data), color="red")
    plt.axhline(np.nanmedian(data), color="blue")
    if x_label: plt.xlabel(x_label)
    if y_label: plt.ylabel(y_label)

    if len(data_labels):
        for i in range(len(data_labels)):
            ax.annotate(data_labels.iloc[i], (axis[i], data[i]), fontsize=5)
