from fcmeans import FCM
from matplotlib import pyplot as plt
from seaborn import scatterplot as scatter
import numpy as np
from datetime import datetime

from algorithms.hc_means import HCM
from datasets.datasets import LineOutlierDataSet, CrossDataSet, MoonDataSet, RingsDataSet, ZigZagOutliersDataSet, \
    NoiseOutliersDataSet
from quantifiers.quantifiers import XieBieni, FukuyamaSugeno

DATASETS = (
    LineOutlierDataSet,
    CrossDataSet,
    MoonDataSet,
    RingsDataSet,
    ZigZagOutliersDataSet,
    NoiseOutliersDataSet,
    LineOutlierDataSet
)

N_CLUSTERS = (2, 3, 4)


# todo refactor into a class
def fcm_example():

    for ds_class in DATASETS:
        xb_results = []
        fs_results = []

        for n_clusters in N_CLUSTERS:
            dataset = ds_class()
            points_matrix = np.concatenate((dataset.x.reshape(-1, 1), dataset.y.reshape(-1, 1)), axis=1)

            fcm = FCM(n_clusters=n_clusters)
            fcm.fit(points_matrix)

            centers = fcm.centers
            labels = fcm.u.argmax(axis=1)

            xb = XieBieni(points_matrix, centers, 2.0)
            fs = FukuyamaSugeno(points_matrix, centers, 2.0)
            xb_results.append(xb.calculate())
            fs_results.append(fs.calculate())

            show_results(dataset, centers, labels)

        plt.subplot(1, 2, 1)
        plt.plot(N_CLUSTERS, xb_results)
        plt.subplot(1, 2, 2)
        plt.plot(N_CLUSTERS, fs_results)


def hcm_example():

    for ds_class in DATASETS:
        xb_results = []
        fs_results = []

        for n_clusters in N_CLUSTERS:
            dataset = ds_class()

            points_matrix = np.concatenate((dataset.x.reshape(-1, 1), dataset.y.reshape(-1, 1)), axis=1)

            hcm = HCM(n_clusters=n_clusters)
            hcm.fit(dataset.x, dataset.y)

            centers = np.array(hcm.centroids)
            labels = hcm.matrix[:, 2]

            xb = XieBieni(points_matrix, centers, 2.0)
            fs = FukuyamaSugeno(points_matrix, centers, 2.0)
            xb_results.append(xb.calculate())
            fs_results.append(fs.calculate())

            show_results(dataset, centers, labels)

        plt.subplot(1, 2, 1)
        plt.plot(N_CLUSTERS, xb_results)
        plt.subplot(1, 2, 2)
        plt.plot(N_CLUSTERS, fs_results)


def show_results(dataset, centers, labels, save=False):
    f, axes = plt.subplots(1, 2, figsize=(11, 5))

    scatter(dataset.x, dataset.y, ax=axes[0])
    scatter(dataset.x, dataset.y, ax=axes[1], hue=labels)
    scatter(centers[:, 0], centers[:, 1], ax=axes[1], marker="s", s=200)

    plt.show()

    if save:
        plt.savefig(f'{dataset.__class__}{datetime.now().microsecond}.png', dpi=300)


if __name__ == "__main__":

    fcm_example()

    hcm_example()
