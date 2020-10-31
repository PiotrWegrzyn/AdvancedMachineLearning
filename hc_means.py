import math

from fcmeans import FCM
from sklearn.datasets import make_blobs
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


class HCM:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.centroids = []
        self.matrix = None
        self.change_made = True

    def set_random_centers(self):
        random_rows = np.random.randint(len(self.matrix), size=self.n_clusters)
        self.centroids = []
        for row_index in random_rows:
            x = self.matrix[row_index][0]
            y = self.matrix[row_index][1]
            self.centroids.append((x, y))

    def fit(self, data_x, data_y):
        data_group = np.array(data_y)
        self.matrix = np.concatenate((data_x.reshape(-1, 1), data_y.reshape(-1, 1), data_group.reshape(-1, 1)), axis=1)

        self.set_random_centers()

        self.change_made = True

        while self.change_made:
            self.change_made = False
            self.assign_to_groups()
            self.calculate_group_centroids()

    def assign_to_groups(self):

        for i, row in enumerate(self.matrix):
            x, y, g = row[0], row[1], row[2]
            new_group = self.assign_group(x, y)
            if g != new_group:
                self.change_made = True
                self.matrix[i][2] = new_group

    def assign_group(self, x, y):
        closest = None
        group = None
        for group_id, center in enumerate(self.centroids):
            distance = self.calcualte_distance(x, y, center[0], center[1])

            if closest is None or distance < closest:
                closest = distance
                group = group_id

        return group

    def calcualte_distance(self, point_x, point_y, center_x, center_y):
        return math.sqrt(math.sqrt((point_x - center_x) ** 2) + math.sqrt((point_y - center_y) ** 2))

    def calculate_group_centroids(self):
        for g in range(self.n_clusters):
            group = self.matrix[self.matrix[:, 2] == g]
            x = group[:, 0]
            y = group[:, 1]
            group_centroid_x = np.mean(x)
            group_centroid_y = np.mean(y)
            self.centroids[g] = (group_centroid_x, group_centroid_y)


for ds_class in datasets:

    dataset = ds_class()

    for n_clusters in (2, 3, 4):
        hcm = HCM(n_clusters=n_clusters)
        hcm.fit(dataset.x, dataset.y)

        fcm_centers = np.array(hcm.centroids)
        fcm_labels = hcm.matrix[:, 2]

        f, axes = plt.subplots(1, 2, figsize=(11, 5))
        scatter(dataset.x, dataset.y, ax=axes[0])
        scatter(dataset.x, dataset.y, ax=axes[1], hue=fcm_labels)
        scatter(fcm_centers[:, 0], fcm_centers[:, 1], ax=axes[1], marker="s", s=200)
        plt.show()
