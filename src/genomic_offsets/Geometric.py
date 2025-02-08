"""Geometric genomic offset"""

import numpy as np
from numba import njit
from .RidgeLFMM import RidgeLFMM


@njit
def genetic_gap(
    covariance: np.ndarray,
    environmental_factors: np.ndarray,
    target_environmental_factors: np.ndarray,
) -> np.ndarray:
    offsets = np.zeros(environmental_factors.shape[0])
    for i in range(len(offsets)):
        diff = environmental_factors[i, :] - target_environmental_factors[i, :]
        offsets[i] = np.dot(np.dot(diff, covariance), diff.T)
    return offsets


class GeometricGO:
    """Geometric genomic offset statistic."""

    def __init__(
        self, n_latent_factors: int, lambda_: float  # Number of latent factors
    ):  # Regularization parameter
        self.K = n_latent_factors
        self.lambda_ = lambda_
        self._LFMM = None
        self._mx = None
        self._sx = None
        self.Cb = None

    def __str__(self):
        return f"Geometric genomic offset with K={self.K} and lambda={self.lambda_}"

    __repr__ = __str__

    def fit(
        self,
        genotypes: np.ndarray,  # Genotype matrix (nxL)
        environmental_factors: np.ndarray,
    ):  # Environmental matrix (nxP)
        """
        Fits the Geometric genomic offset model.
        :param genotypes: 2D array of genotypes (nxL).
        :param environmental_factors: 2D array of environmental factors (nxP).
        :return:
        """
        n1, l = genotypes.shape
        n2, p = environmental_factors.shape
        if n1 != n2:
            raise ValueError("Dimensions of array don't match")
        # Scale data and save it to predict later
        genotypes = genotypes - np.mean(genotypes, axis=0)
        mx = np.mean(environmental_factors, axis=0)
        environmental_factors = environmental_factors - mx
        sx = np.std(environmental_factors, axis=0)
        environmental_factors = environmental_factors / sx
        self._mx = mx
        self._sx = sx
        model = self._LFMM = RidgeLFMM(n_latent_factors=self.K, lambda_=self.lambda_)
        model.fit(genotypes=genotypes, environmental_factors=environmental_factors)
        self.Cb = np.dot(model.B.T, model.B) / model.B.shape[0]

    def _rescale_env(
        self,
        environmental_factors: np.ndarray,  # Environmental matrix (nxP)
    ) -> np.ndarray:  # Re-scaled environmental matrix
        if self._mx is None or self._sx is None:
            raise ValueError("You have to fit the model first!")
        return (environmental_factors - self._mx) / self._sx

    def predict(
        self, environmental_factors: np.ndarray  # Environmental matrix (nxP)
    ) -> np.ndarray:  # Predicted allele frequencies
        """
        Predicts the *centered* optimal genotypes for the fitted environmental matrix.
        :param environmental_factors: 2D array of environmental factors (nxP)
        :return: 2D array of centered predicted genotypes (nxL)
        """
        environmental_factors = self._rescale_env(environmental_factors)
        return self._LFMM.predict(environmental_factors)

    def genomic_offset(
        self,
        environmental_factors: np.ndarray,  # Environmental matrix (nxP)
        target_environmental_factors: np.ndarray,  # Altered environmental matrix (nxP)
    ) -> np.ndarray:  # A vector of genomic offsets (n)
        """
        Calculates the geometric genomic offset statistic.
        :param environmental_factors: 2D array of environmental factors (nxP)
        :param target_environmental_factors: 2D array of target environmental factors (nxP)
        :return: Array of genomic offset statistics
        """
        if environmental_factors.shape != target_environmental_factors.shape:
            raise ValueError("Dimensions of array don't match")
        return genetic_gap(
            self.Cb,
            self._rescale_env(environmental_factors),
            self._rescale_env(target_environmental_factors),
        )
