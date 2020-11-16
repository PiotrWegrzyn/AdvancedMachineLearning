import random

import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.datasets import make_moons
from datetime import datetime

datasets = []


class DataSet:
    BIG_SAMPLE_SIZE = 1000
    SMALL_SAMPLE_SIZE = 100
    DISTORTION = 0.3

    def __init__(self):
        self.x, self.y, self.z = self.generate()

    def generate(self):
        raise NotImplementedError

    def show(self, x=None, y=None, z=None, dimensions=2, save=False):
        if not x:
            x = self.x.reshape(-1, 1)
        if not y:
            y = self.y.reshape(-1, 1)
        if not z and dimensions == 3:
            z = self.z.reshape(-1, 1)

        self.scatter(x, y, z)
        plt.grid()
        plt.show()

        if save:
            plt.savefig(f'{self.__class__}{datetime.now().microsecond}.png', dpi=300)

    @staticmethod
    def scatter(x, y, z=None):
        if z is None:
            plt.scatter(x, y)
        else:
            ax = plt.axes(projection="3d")

            ax.scatter3D(x, y, z, color="green")

    def generate_3d(self, coords):
        z_coords = [self.distorted_random_number() for _ in range(len(coords))]

        return np.array(z_coords)

    def distorted_random_number(self):
        return random.uniform(-2, 2) + np.random.normal(0, self.DISTORTION)

    # old method for generation based on other dimention
    # def generate_3d(self, coords):
        # # z_coords = np.linspace(-2, 2, points)
        # z_coords = [coord + np.random.normal(0, self.DISTORTION) for coord in coords]
        # return np.random.choice(z_coords, len(coords))


class CrossDataSet(DataSet):

    def generate(self):
        random_coords = np.linspace(-2, 2, self.BIG_SAMPLE_SIZE * 10)
        x_coords = np.random.choice(random_coords, self.BIG_SAMPLE_SIZE)
        z_coords = np.random.choice(random_coords, self.BIG_SAMPLE_SIZE)

        half_size = int(self.BIG_SAMPLE_SIZE / 2)

        y_line1 = [x + np.random.normal(0, self.DISTORTION) for x in x_coords[:half_size]]
        y_line2 = [x * (-1) + np.random.normal(0, self.DISTORTION) for x in x_coords[half_size:]]

        self.x = np.array(x_coords)
        self.z = np.array(z_coords)
        self.y = np.append(y_line1, y_line2)

        return self.x, self.y, self.z


class MoonDataSet(DataSet):

    def generate(self):
        coords, classes = make_moons(n_samples=self.BIG_SAMPLE_SIZE, noise=self.DISTORTION/6)

        return coords[:, 0], coords[:, 1], self.generate_3d(coords[:, 0])


class RingsDataSet(DataSet):
    DISTORTION = DataSet.DISTORTION/5

    def __init__(self, radii=(2, 0.5)):
        self.radii = radii
        super().__init__()

    def generate(self):
        alphas = np.random.random(size=self.BIG_SAMPLE_SIZE*10)
        alphas = np.multiply(alphas, 2 * math.pi)

        x_coords = []
        y_coords = []
        points = 0
        for radius in self.radii:
            circle_size_ratio = radius / sum(self.radii)
            circle_points = int(self.BIG_SAMPLE_SIZE*circle_size_ratio)
            points += circle_points

            alpha = np.random.choice(alphas, circle_points)

            x = [(radius + (np.random.normal(0, self.DISTORTION))) * math.cos(ang) for ang in alpha]
            y = [(radius + (np.random.normal(0, self.DISTORTION))) * math.sin(ang) for ang in alpha]
            x_coords.extend(x)
            y_coords.extend(y)

        return np.array(x_coords), np.array(y_coords), self.generate_3d(x_coords)


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

        return x, y, self.generate_3d(x)


class NoiseOutliersDataSet(DataSet):

    def __init__(self, radii=(2, 0.5)):
        self.radii = radii
        super().__init__()

    def generate(self):
        x = np.linspace(-1.5, 4, int(self.SMALL_SAMPLE_SIZE/2))
        x2 = np.linspace(-1.5, -0.5, int(self.SMALL_SAMPLE_SIZE/2))
        y = np.random.random(int(self.SMALL_SAMPLE_SIZE/2)) * 5 - 3.5
        y2 = np.random.random(int(self.SMALL_SAMPLE_SIZE/2)) + 0.5

        x_coords = np.append(x, x2)
        y_cords = np.append(y, y2)

        return x_coords, y_cords, self.generate_3d(x_coords)


class LineOutlierDataSet(DataSet):

    def generate(self):
        x = np.linspace(-2, 2, self.SMALL_SAMPLE_SIZE)
        y = [wx + 0.25 + (np.random.random() - 0.5) / 2 for wx in x]

        x2 = np.linspace(0, 0.5, self.SMALL_SAMPLE_SIZE)
        y2 = (np.random.random(self.SMALL_SAMPLE_SIZE) - 2) / 2

        x_coords = np.append(x, x2)
        y_cords = np.append(y, y2)

        return x_coords, y_cords, self.generate_3d(x_coords)

