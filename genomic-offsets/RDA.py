"""Redundancy analysis genomic offset"""
import numpy as np
import statsmodels.api as sm
class RDA:
    """Redundancy analysis genomic offset statistic."""

    def __init__(self, n_latent_factors: int):
        self._reg = None
        self._pca = None
        self.n_latent_factors = n_latent_factors

    def __str__(self):
        return f"Redundancy analysis model with K={self.n_latent_factors}"

    __repr__ = __str__
    def fit(self,
            allele_frequency: np.ndarray,  # Allele frequency matrix (nxL)
            environmental_factors: np.ndarray):  # Environmental matrix (nxP)
        """Fits the RDA model. """
        n1, L = allele_frequency.shape
        n2, p = environmental_factors.shape
        if n1 != n2:
            raise ValueError("Dimensions of array don't match")
        environmental_factors = sm.add_constant(environmental_factors)
        model = sm.OLS(allele_frequency, environmental_factors)
        self._reg = model.fit()
        self._pca = sm.PCA(self._reg.predict(environmental_factors), ncomp=self.n_latent_factors, method="nipals")
    def predict(self,
                environmental_factors: np.ndarray  # Environmental matrix (nxP)
                ) -> np.ndarray:  # Predicted allele frequencies
        """Predicts the projected alleles for a given environmental matrix."""
        environmental_factors = sm.add_constant(environmental_factors)
        return np.dot(self._reg.predict(environmental_factors), self._pca.loadings)

    def genomic_offset(self,
                       environmental_factors: np.ndarray,  # Environmental matrix (nxP)
                       target_environmental_factors: np.ndarray,  # Altered environmental matrix (nxP)
                       ) -> np.ndarray:  # A vector of genomic offsets (n)
        """Calculates the genomic offset statistic. """
        if environmental_factors.shape != target_environmental_factors.shape:
            raise ValueError("Dimensions of array don't match")
        offset = np.zeros(environmental_factors.shape[0])
        diff = self.predict(environmental_factors) - self.predict(target_environmental_factors)
        weights = self._pca.eigenvals / np.sum(self._pca.eigenvals)
        L = self._reg.params.shape[1]
        for distance, w in zip(diff.T, weights):
            offset += distance ** 2 * w / L
        return offset


class PartialRDA:
    """Partial redundancy analysis genomic offset statistic."""

    def __init__(self, n_latent_factors: int):
        self._mod = None
        self._reg_X = None
        self._reg_residuals = None
        self._pca = None
        self.n_latent_factors = n_latent_factors

    def __str__(self):
        return f"Partial redundancy analysis model with K={self.n_latent_factors}"

    __repr__ = __str__

    def fit(self,
            allele_frequency: np.ndarray,  # Allele frequency matrix (nxL)
            environmental_factors: np.ndarray,  # Environmental matrix (nxP)
            covariates: np.ndarray):  # Covariate matrix (nxZ)
        """Fits the partial RDA model. """
        n1, L = allele_frequency.shape
        n2, P = environmental_factors.shape
        n3, Z = covariates.shape
        if n1 != n2 or n1 != n3:
            raise ValueError("Dimensions of array don't match")
        covariates = sm.add_constant(covariates)
        self._reg_X = sm.OLS(environmental_factors, covariates).fit()
        residuals_y = sm.OLS(allele_frequency, covariates).fit().resid.reshape(allele_frequency.shape)
        residuals_x = sm.add_constant(self._reg_X.resid.reshape(environmental_factors.shape))
        self._reg_residuals = sm.OLS(residuals_y, residuals_x).fit()
        self._pca = sm.PCA(self._reg_residuals.predict(residuals_x), ncomp=self.n_latent_factors, method="nipals")

    def predict(self,
                environmental_factors: np.ndarray,  # Environmental matrix (nxP)
                covariates: np.ndarray  # Covariate matrix (nxZ)
                ) -> np.ndarray:  # Predicted allele frequencies
        """Predicts the projected alleles for a given environmental matrix."""
        covariates = sm.add_constant(covariates)
        residuals_x = (environmental_factors - self._reg_X.predict(covariates).reshape(environmental_factors.shape))
        residuals_x = sm.add_constant(residuals_x)
        pred_residuals_y = self._reg_residuals.predict(residuals_x)
        return np.dot(pred_residuals_y, self._pca.loadings)

    def genomic_offset(self,
                       environmental_factors: np.ndarray,  # Environmental matrix (nxP)
                       target_environmental_factors: np.ndarray,  # Altered environmental matrix (nxP)
                       covariates: np.ndarray  # Covariate matrix (nxZ)
                       ) -> np.ndarray:  # A vector of genomic offsets (n)
        """Calculates the genomic offset statistic. """
        if environmental_factors.shape != target_environmental_factors.shape:
            raise ValueError("Dimensions of array don't match")
        if environmental_factors.shape[0] != covariates.shape[0]:
            raise ValueError("Dimensions of array don't match")
        offset = np.zeros(environmental_factors.shape[0])
        diff = self.predict(environmental_factors, covariates) - self.predict(target_environmental_factors, covariates)
        weights = self._pca.eigenvals / np.sum(self._pca.eigenvals)
        L = self._reg_residuals.params.shape[1]
        for distance, w in zip(diff.T, weights):
            offset += distance ** 2 * w / L
        return offset