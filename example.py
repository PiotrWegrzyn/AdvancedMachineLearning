from fcmeans import FCM
from matplotlib import pyplot as plt
from seaborn import scatterplot as scatter
import numpy as np

from algorithms.hc_means import HCM
from datasets.datasets import LineOutlierDataSet, CrossDataSet, MoonDataSet, RingsDataSet, ZigZagOutliersDataSet, \
    NoiseOutliersDataSet

DATASETS = (
    LineOutlierDataSet,
    CrossDataSet,
    MoonDataSet,
    RingsDataSet,
    ZigZagOutliersDataSet,
    NoiseOutliersDataSet,
    LineOutlierDataSet
)


def fcm_example(n_clusters):
    for ds_class in DATASETS:

        dataset = ds_class()

        fcm = FCM(n_clusters=n_clusters)
        fcm.fit(np.concatenate((dataset.x.reshape(-1, 1), dataset.y.reshape(-1, 1)), axis=1))

        centers = fcm.centers
        labels = fcm.u.argmax(axis=1)

        show_results(dataset, centers, labels)


def hcm_example(n_clusters):
    for ds_class in DATASETS:

        dataset = ds_class()

        hcm = HCM(n_clusters=n_clusters)
        hcm.fit(dataset.x, dataset.y)

        centers = np.array(hcm.centroids)
        labels = hcm.matrix[:, 2]

        show_results(dataset, centers, labels)


def show_results(dataset, centers, labels):
    f, axes = plt.subplots(1, 2, figsize=(11, 5))
    scatter(dataset.x, dataset.y, ax=axes[0])
    scatter(dataset.x, dataset.y, ax=axes[1], hue=labels)
    scatter(centers[:, 0], centers[:, 1], ax=axes[1], marker="s", s=200)
    plt.show()


if __name__ == "__main__":
    N_CLUSTERS = (2, 3, 4)

    for n_clusters in N_CLUSTERS:
        fcm_example(n_clusters)

    for n_clusters in N_CLUSTERS:
        hcm_example(n_clusters)
