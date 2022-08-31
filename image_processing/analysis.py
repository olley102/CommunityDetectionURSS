import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples


def get_classes(x, y, return_dict=False):
    """
    Split x into pre-images of y.
    :param x: data.
    :param y: labels.
    :param return_dict: if True, then dict of form {label: x[y==label]} is returned.
    :return: list of split data or dict.
    """

    if isinstance(y[0], collections.Hashable):
        y_hashable = y
    else:
        y_hashable = [tuple(label) for label in y]

    label_name_space = sorted(list(set(y_hashable)))  # cannot make set of y unless hashable. Fix made above

    label_space = np.array([hash(name) for name in label_name_space])
    y_hashed = np.array([hash(name) for name in y_hashable])

    x_np = np.array(x)

    if return_dict:
        classes = {label_name_space[i]: x_np[y_hashed == label] for i, label in enumerate(label_space)}
    else:
        classes = [x_np[y_hashed == label] for i, label in enumerate(label_space)]
    return classes


def silhouette_plot(x, y, figsize=(16, 9), xlim=(-0.1, 1)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    ax.set_xlim(xlim)
    cluster_labels_hash = np.array([hash(label) for label in y])
    silhouette_avg = silhouette_score(x, cluster_labels_hash)
    sample_silhouette_values = silhouette_samples(x, cluster_labels_hash)

    y_lower = 10
    cluster_labels_idx = get_classes(range(len(y)), y, return_dict=True)

    for i, lab in enumerate(cluster_labels_idx):
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels_idx[lab]]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / len(cluster_labels_idx))
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color,
                         edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster names at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(lab))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks(np.arange(int(10 * xlim[0]), int(10 * xlim[1]) + 1) / 10)

    return fig


def performance_density_plot(image1, image2, vmin, vmax, num_bins, measure=None):
    if measure is None:
        def measure(x, y):
            return (x - y) ** 2

    image1_space = np.linspace(vmin, vmax, num_bins)
    interval = image1_space[1] - image1_space[0]
    quality = np.zeros(num_bins)

    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            value = image1[i, j]
            index = round(value / interval)
            meas = measure(value, image2[i, j])
            quality[index] += meas

    quality /= interval

    return image1_space, quality
