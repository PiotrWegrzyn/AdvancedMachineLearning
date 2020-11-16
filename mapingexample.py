from matplotlib import pyplot as plt
from minisom import MiniSom
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler

from datasets.datasets import LineOutlierDataSet, CrossDataSet, MoonDataSet, RingsDataSet, ZigZagOutliersDataSet, \
    NoiseOutliersDataSet


class MappingExample:

    DATASETS = (
        CrossDataSet,
        MoonDataSet,
        RingsDataSet,
        ZigZagOutliersDataSet,
        NoiseOutliersDataSet,
        LineOutlierDataSet,
    )

    def run(self):

        for ds_class in self.DATASETS:

            dataset = ds_class()
            dataset.show(dimensions=3)
            data = dataset.get_data_3d()

            som_shape = (1, 5)
            self.doSOMmagic(data, som_shape)

        data = pd.read_csv("./datasets/wine.data").to_numpy()
        pd_data = load_wine()
        data = data[:, 1:]
        som_shape = (1, 3)

        self.doSOMmagic(data, som_shape)
        self.doMDSmagic(pd_data)

        data = pd.read_csv("./datasets/glass.data").to_numpy()
        data = data[:, 1:-2]
        som_shape = (1, 7)

        self.doSOMmagic(data, som_shape)

        data = load_breast_cancer()
        som_shape = (1, 2)

        self.doSOMmagic(data['data'], som_shape)
        self.doMDSmagic(data)

    def group_data(self, dataset, n_clusters):
        raise NotImplementedError

    def doSOMmagic(self, data, som_shape):
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        som = MiniSom(som_shape[0], som_shape[1], data.shape[1], sigma=.5, learning_rate=.5,
                      neighborhood_function='gaussian', random_seed=10)
        som.pca_weights_init(data)
        som.train_batch(data, 1000, verbose=True)

        winner_coordinates = np.array([som.winner(x) for x in data]).T
        cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)

        # plotting the clusters using the first 2 dimentions of the data
        for c in np.unique(cluster_index):
            plt.scatter(data[cluster_index == c, 0],
                        data[cluster_index == c, 1], label='cluster=' + str(c), alpha=.7)

        # plotting centroids
        for centroid in som.get_weights():
            plt.scatter(centroid[:, 0], centroid[:, 1], marker='x',
                        s=15, linewidths=15, color='k', label='centroid')

        plt.legend()
        plt.show()

        plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
        plt.colorbar()
        plt.show()

    def doMDSmagic(self, data_pd):
        data = data_pd['data']
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(data)
        mds = MDS(2)
        X_2d = mds.fit_transform(X_scaled)

        colors = ['red', 'green', 'blue']
        plt.rcParams['figure.figsize'] = [7, 7]
        plt.rc('font', size=14)
        for i in np.unique(data_pd.target):
            subset = X_2d[data_pd.target == i]

            x = [row[0] for row in subset]
            y = [row[1] for row in subset]
            plt.scatter(x, y, c=colors[i], label=data_pd.target_names[i])
            plt.legend()
        plt.show()


if __name__ == "__main__":

    # Project 02
    mapping_example = MappingExample()
    mapping_example.run()
