from fcmeans import FCM
from matplotlib import pyplot as plt
from seaborn import scatterplot as scatter
import numpy as np

from datasets.datasets import LineOutlierDataSet, CrossDataSet, MoonDataSet, RingsDataSet, ZigZagOutliersDataSet, \
    NoiseOutliersDataSet


datasets = (
    LineOutlierDataSet,
    CrossDataSet,
    MoonDataSet,
    RingsDataSet,
    ZigZagOutliersDataSet,
    NoiseOutliersDataSet,
    LineOutlierDataSet
)

for ds_class in datasets:

    dataset = ds_class()

    for n_clusters in (2, 3, 4):
        fcm = FCM(n_clusters=n_clusters)
        fcm.fit(np.concatenate((dataset.x.reshape(-1, 1), dataset.y.reshape(-1, 1)), axis=1))

        fcm_centers = fcm.centers
        fcm_labels = fcm.u.argmax(axis=1)

        f, axes = plt.subplots(1, 2, figsize=(11, 5))
        scatter(dataset.x, dataset.y, ax=axes[0])
        scatter(dataset.x, dataset.y, ax=axes[1], hue=fcm_labels)
        scatter(fcm_centers[:, 0], fcm_centers[:, 1], ax=axes[1], marker="s", s=200)
        plt.show()