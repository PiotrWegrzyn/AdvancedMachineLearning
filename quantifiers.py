import numpy as np
import scipy.spatial


class Quantifier(object):
    def __init__(self, x, v, m):
        self.x = x
        self.v = v
        self.m = m
        self.u = self.fcm_get_u(x, v, m)

    @classmethod
    def fcm_get_u(cls, x, v, m):
        distances = cls.pairwise_squared_distances(x, v)
        nonzero_distances = np.fmax(distances, np.finfo(np.float64).eps)
        inv_distances = np.reciprocal(nonzero_distances) ** (1 / (m - 1))
        return inv_distances.T / np.sum(inv_distances, axis=1)

    @staticmethod
    def calculate_covariances(x, u, v, m):
        c, n = u.shape
        d = v.shape[1]

        um = u ** m

        covariances = np.zeros((c, d, d))

        for i in range(c):
            xv = x - v[i]
            uxv = um[i, :, np.newaxis] * xv
            covariances[i] = np.einsum('ni,nj->ij', uxv, xv) / np.sum(um[i])

        return covariances


    @staticmethod
    def pairwise_squared_distances(A, B):
        return scipy.spatial.distance.cdist(A, B) ** 2


class XieBieni(Quantifier):
    target = 'min'

    def calculate(self):
        n = self.x.shape[0]
        c = self.v.shape[0]

        um = self.u ** self.m

        d2 = self.pairwise_squared_distances(self.x, self.v)
        v2 = self.pairwise_squared_distances(self.v, self.v)

        v2[v2 == 0.0] = np.inf

        return np.sum(um.T * d2) / (n * np.min(v2))


class FukuyamaSugeno(Quantifier):
    target = 'min'

    def calculate(self):

        um = self.u ** self.m

        v_mean = self.v.mean(axis=0)

        d2 = self.pairwise_squared_distances(self.x, self.v)

        distance_v_mean_squared = np.linalg.norm(self.v - v_mean, axis=1, keepdims=True) ** 2

        return np.sum(um.T * d2) - np.sum(um * distance_v_mean_squared)
