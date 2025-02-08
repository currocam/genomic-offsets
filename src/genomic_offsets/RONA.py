"""Risk of non-adaptedness"""
import numpy as np
import statsmodels.api as sm
class RONA:
    """Risk of non-adaptedness genomic offset statistic."""
    def __init__(self):
        self._reg = None
    def __str__(self):
        return "RONA model."
    __repr__ = __str__
    def fit(self,
            allele_frequency: np.ndarray,  # Allele frequency matrix (nxL)
            environmental_factors: np.ndarray): # Environmental matrix (nxP)
        """Fits the RONA model. """
        n1, l = allele_frequency.shape
        n2, p = environmental_factors.shape
        if n1 != n2:
            raise ValueError("Dimensions of array don't match")
        environmental_factors = sm.add_constant(environmental_factors)
        model = sm.OLS(allele_frequency, environmental_factors)
        self._reg = model.fit()

    def predict(self,
                environmental_factors: np.ndarray  # Environmental matrix (nxP)
                )-> np.ndarray: # Predicted allele frequencies
        """Predicts the allele frequencies for a given environmental matrix. """
        return self._reg.predict(sm.add_constant(environmental_factors))
    def genomic_offset(self,
                       environmental_factors: np.ndarray,  # Environmental matrix (nxP)
                       target_environmental_factors: np.ndarray,  # Altered environmental matrix (nxP)
                       )-> np.ndarray: # A vector of genomic offsets (n)
        """Calculates the genomic offset statistics."""
        l = self._reg.params.shape[1]
        environmental_factors, target_environmental_factors = sm.add_constant(environmental_factors), sm.add_constant(target_environmental_factors)
        if environmental_factors.shape != target_environmental_factors.shape:
            raise ValueError("Dimensions of array don't match")
        return np.sum(np.abs(self._reg.predict(environmental_factors) - self._reg.predict(target_environmental_factors)), axis=1) / l