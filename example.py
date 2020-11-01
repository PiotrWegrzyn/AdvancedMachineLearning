from fcmeans import FCM
from matplotlib import pyplot as plt
from seaborn import scatterplot as scatter, lineplot
import numpy as np
from datetime import datetime

from algorithms.hc_means import HCM
from datasets.datasets import LineOutlierDataSet, CrossDataSet, MoonDataSet, RingsDataSet, ZigZagOutliersDataSet, \
    NoiseOutliersDataSet
from quantifiers.quantifiers import XieBieni, FukuyamaSugeno


class Example:
    N_CLUSTERS = (2, 3, 4)

    DATASETS = (
        LineOutlierDataSet,
        CrossDataSet,
        MoonDataSet,
        RingsDataSet,
        ZigZagOutliersDataSet,
        NoiseOutliersDataSet,
        LineOutlierDataSet
    )

    def run(self):

        for ds_class in self.DATASETS:
            xb_results = []
            fs_results = []

            f, axes = plt.subplots(3, 2, figsize=(11, 5), squeeze=True)

            dataset = ds_class()
            points_matrix = np.concatenate((dataset.x.reshape(-1, 1), dataset.y.reshape(-1, 1)), axis=1)

            scatter(x=dataset.x, y=dataset.y, ax=axes[0][0])

            for i, n_clusters in enumerate(self.N_CLUSTERS):
                centers, labels = self.group_data(dataset, n_clusters)

                xb = XieBieni(points_matrix, centers, 2.0)
                xb_results.append(xb.calculate())

                fs = FukuyamaSugeno(points_matrix, centers, 2.0)
                fs_results.append(fs.calculate())

                row = int((i+1) / 2)
                col = (i+1) % 2
                scatter(x=dataset.x, y=dataset.y, ax=axes[row][col], hue=labels)
                scatter(x=centers[:, 0], y=centers[:, 1], ax=axes[row][col], marker="s", s=200)

            axes[2][0].set_title("Xie Bieni")
            lineplot(x=self.N_CLUSTERS, y=xb_results, ax=axes[2][0])

            axes[2][1].set_title("Fukuyama-Sugeno")
            lineplot(x=self.N_CLUSTERS, y=fs_results, ax=axes[2][1])

            plt.show()

    def group_data(self, dataset, n_clusters):
        raise NotImplementedError


class FCMExample(Example):
    def group_data(self, dataset, n_clusters):
        points_matrix = np.concatenate((dataset.x.reshape(-1, 1), dataset.y.reshape(-1, 1)), axis=1)

        fcm = FCM(n_clusters=n_clusters)
        fcm.fit(points_matrix)

        centers = fcm.centers
        labels = fcm.u.argmax(axis=1)

        return centers, labels


class HCMExample(Example):
    def group_data(self, dataset, n_clusters):
        hcm = HCM(n_clusters=n_clusters)
        hcm.fit(dataset.x, dataset.y)

        centers = np.array(hcm.centroids)
        labels = hcm.matrix[:, 2]

        return centers, labels


if __name__ == "__main__":

    hcm_example = HCMExample()
    hcm_example.run()

    fcm_example = FCMExample()
    fcm_example.run()
