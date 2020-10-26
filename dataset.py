import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
import matplotlib
from fcmeans import FCM

datasets = []


class DataSet:
    BIG_SAMPLE_SIZE = 100
    SMALL_SAMPLE_SIZE = 50

    def __init__(self):
        self.x, self.y = self.generate()

    def generate(self):
        raise NotImplementedError

    def show(self, x=None, y=None):
        if not x:
            x = self.x
        if not y:
            y = self.y

        self.scatter(x, y)
        plt.grid()
        plt.show()

    @staticmethod
    def scatter(x, y):
        plt.scatter(x, y)


class CrossDataSet(DataSet):

    def generate(self):
        random_x_coords = np.linspace(-2, 2, self.BIG_SAMPLE_SIZE * 10)
        chosen_x_coords = np.random.choice(random_x_coords, self.BIG_SAMPLE_SIZE)

        half_size = int(self.BIG_SAMPLE_SIZE/2)

        y_line1 = [x + np.random.normal(0, 0.25) for x in chosen_x_coords[:half_size]]
        y_line2 = [x * (-1) + np.random.normal(0, 0.3) for x in chosen_x_coords[half_size:]]

        self.x = np.append(chosen_x_coords[:half_size], chosen_x_coords[half_size:])
        self.y = np.append(y_line1, y_line2)

        return self.x, self.y


cross_ds = CrossDataSet()
cross_ds.show()
datasets.append(cross_ds)


class MoonDataSet(DataSet):

    def generate(self):
        coords, classes = make_moons(n_samples=self.BIG_SAMPLE_SIZE, noise=0.05)

        return coords[:, 0], coords[:, 1]


moon_ds = MoonDataSet()
moon_ds.show()
datasets.append(moon_ds)


class RingsDataSet(DataSet):

    def __init__(self, radii=(2, 0.5)):
        self.radii = radii
        super().__init__()

    def generate(self):
        alphas = np.random.random(size=self.BIG_SAMPLE_SIZE*10)
        alphas = np.multiply(alphas, 2 * math.pi)

        x_coords = []
        y_coords = []
        for radius in self.radii:
            alpha = np.random.choice(alphas, int(self.BIG_SAMPLE_SIZE / len(self.radii)))
            x = [(radius + (np.random.normal(0, 0.3)) / 4) * math.cos(ang) for ang in alpha]
            y = [(radius + (np.random.normal(0, 0.3)) / 4) * math.sin(ang) for ang in alpha]
            x_coords.extend(x)
            y_coords.extend(y)

        return np.array(x_coords), np.array(y_coords)


rings_ds = RingsDataSet()
rings_ds.show()
datasets.append(rings_ds)


class ZigZagOutliersDataSet(DataSet):

    def generate(self):
        x = np.linspace(-2, 2, 150)

        x1 = np.linspace(-1, -0.5, 50)
        y1 = np.random.random(50) - 1.5

        x2 = np.linspace(0.5, 1, 50)
        y2 = np.random.random(50) + 0.5

        y3 = x[:50] * 3.5 + 4.6

        y3 = np.append(y3, x[50:100] * -3.5)

        y3 = np.append(y3, x[100:] * 3.5 - 4.6)
        y3 = [wx + (np.random.random() - 0.5) / 2 for wx in y3]

        x = np.append(np.append(x1, x2), x)
        y = np.append(np.append(y1, y2), y3)

        return x, y



zz_ds = ZigZagOutliersDataSet()
zz_ds.show()
datasets.append(zz_ds)


class NoiseOutliersDataSet(DataSet):

    def __init__(self, radii=(2, 0.5)):
        self.radii = radii
        super().__init__()

    def generate(self):
        x = np.linspace(-1.5, 4, int(self.SMALL_SAMPLE_SIZE/2))
        x2 = np.linspace(-1.5, -0.5, int(self.SMALL_SAMPLE_SIZE/2))
        y = np.random.random(int(self.SMALL_SAMPLE_SIZE/2)) * 5 - 3.5
        y2 = np.random.random(int(self.SMALL_SAMPLE_SIZE/2)) + 0.5

        return np.append(x, x2), np.append(y, y2)


ns_ds = NoiseOutliersDataSet()
ns_ds.show()
datasets.append(ns_ds)


class LineOutlierDataSet(DataSet):

    def generate(self):
        x = np.linspace(-2, 2)
        y = [wx + 0.25 + (np.random.random() - 0.5) / 2 for wx in x]

        x2 = np.linspace(0, 0.5, self.SMALL_SAMPLE_SIZE)
        y2 = (np.random.random(self.SMALL_SAMPLE_SIZE) - 2) / 2

        return np.append(x, x2), np.append(y, y2)


line_ds = LineOutlierDataSet()
line_ds.show()
datasets.append(line_ds)


# color_tab = ['r', 'g', 'b', 'k', 'p']
#
#
# fig, axes = plt.subplots(6, 3)
# matplotlib.rcParams['figure.figsize'] = [20, 15]
# i = 0
# j = 0
# data = None
# for dataset in datasets:
#     x = dataset.x
#     y = dataset.y
#
#     input_X = np.c_[x, y]
#     for x in range(2, 5):
#         clf = KMeans(n_clusters=x)
#         clf.fit(input_X)
#         lab = clf.labels_
#         cen = clf.cluster_centers_
#         data = np.c_[input_X, lab]
#         col = 0
#         for le in np.unique(lab):
#             dat = data[data[:, 2] == le]
#             axes[i, j].scatter(dat[:, 0], dat[:, 1], color=color_tab[le])
#         axes[i, j].set_title('C=: {}'.format(x))
#         axes[i, j].grid()
#         for centers in cen:
#             axes[i, j].scatter(centers[0], centers[0], marker='x', s=600, linewidth=9, color=color_tab[col])
#             col += 1
#
#         j += 1
#         if j == 3:
#             j = 0
#     i += 1
# plt.tight_layout()
# plt.show()
#
# print(data[:, 2])
