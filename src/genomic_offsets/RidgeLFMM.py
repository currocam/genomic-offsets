"""Ridge solutions to a Latent factor mixed model"""
import numpy as np
from numpy import ndarray
from scipy.stats import f
from scipy.linalg import svd
from .TracyWidom import TracyWidom
from numba import njit

def lfmm2(genotypes, environmental_factors, n_latent_factors, lambda_=1e-5):
    # Assumes data has been already centered
    n, p = environmental_factors.shape
    # Singular Value Decomposition (SVD) of X
    Q, sigma, _ = svd(environmental_factors, overwrite_a=True)
    # Create the diagonal matrix D
    d = np.ones(n)
    d_inv = np.ones(n)
    d[:p] = np.sqrt(lambda_ / (lambda_ + sigma**2))
    d_inv[:p] = np.sqrt((lambda_ + sigma**2) / lambda_)
    D = np.diag(d)
    Dinv = np.diag(d_inv)
    # SVD of the modified Y (D * Q.T * Y)
    U_svd, S_svd, Vt_svd = svd(np.dot(D, np.dot(Q.T, genotypes)), full_matrices=False)
    # Get first K components from SVD
    U = np.dot(np.dot(Q, Dinv), U_svd[:, :n_latent_factors]) * S_svd[:n_latent_factors]
    Vt = Vt_svd[:n_latent_factors, :]
    # Ridge regression to compute Bt
    Bt = np.linalg.inv(np.dot(environmental_factors.T, environmental_factors) + lambda_ * np.eye(p)).dot(environmental_factors.T).dot(genotypes - np.dot(U, Vt))
    return U, Vt.T, Bt.T


@njit
def tracy_widom_statistics(
    eigenvalues: np.ndarray, # Array of eigenvalues
    ) -> np.ndarray: # Array of the Tracy-Widom statistics.
    """
    Compute the Tracy-Widom statistics for a given list of eigenvalues (calculated from the genotype matrix)
    :param eigenvalues: Array of eigenvalues
    :return: Array of Tracy-Widom statistics
    """
    eigenvalues = np.sort(eigenvalues)
    l1 = np.cumsum(eigenvalues)
    l1 = np.flip(l1)  # Reverse the result
    l2 = np.cumsum(eigenvalues ** 2)
    l2 = np.flip(l2)
    n = np.arange(len(eigenvalues), 0, -1)
    s2 = (n ** 2 * l2) / (l1 ** 2)
    v = n * (n + 2) / (s2 - n)
    l = n * np.flip(eigenvalues) / l1
    v_st = np.where(v - 1 > 0, v - 1, np.nan)
    v_st = np.sqrt(v_st)
    n_st = np.sqrt(n)
    mu = ((v_st + n_st) ** 2) / v
    sigma = (v_st + n_st) / v * (1 / v_st + 1 / n_st) ** (1 / 3)
    tw_stats = (l - mu) / sigma
    return tw_stats


def significant_num_factors_tracy_widom(
        tw_stats: np.ndarray, # Tracy-Widom statistics
        thresholds: np.ndarray # Array of significance levels
        ) -> np.ndarray: # Number of significant eigenvalues
    """
    Compute the number of significant factors to include according to the Tracy-Widom distribution.
    :param tw_stats: Array of Tracy-Widom statistics
    :param thresholds: Array of significant levels
    :return: Array of proposed number of significant factors
    """
    n = len(thresholds)
    res = np.zeros(n)
    for i in range(n):
        p_values = 1- TracyWidom(beta=1).cdf(tw_stats)
        indices = np.where(p_values > thresholds[i])[0]
        if len(indices) == 0:
            res[i] = len(p_values)
            pass
        else:
            res[i] = max(indices[0] - 1, 0)
    return res

class RidgeLFMM:
    """
    Ridge solutions to a Latent Factor Mixed Model (LFMM).
    """
    def __init__(self,
                 n_latent_factors: int,  # Number of latent factors
                 lambda_: float): # Regularization parameter
        """
        Create a Ridge LFMM model.
        :param n_latent_factors: Integer number of latent factors
        :param lambda_: Regularization parameter
        """
        self.U, self.V, self.B = None, None, None
        self.K = n_latent_factors
        self.lambda_ = lambda_
    def __str__(self):
        """Returns a string representation of the RidgeLFMM instance."""
        return f"RidgeLFMM model with K={self.K} and lambda={self.lambda_}"
    __repr__ = __str__
    def fit(self,
            genotypes: np.ndarray,  # Genotype matrix (nxL)
            environmental_factors: np.ndarray): # Environmental matrix (nxP)
        """
        Fit the Ridge LFMM model to the given genotypes and environmental factors.
        :param genotypes: 2D array of genotypes (nxL)
        :param environmental_factors: 2D array of environmental factors (nxP)
        :return:
        """
        n1, l = genotypes.shape
        n2, p = environmental_factors.shape
        if n1 != n2:
            raise ValueError("Dimensions of array don't match")
        genotypes = genotypes - np.mean(genotypes, axis=0)
        environmental_factors = environmental_factors - np.mean(environmental_factors, axis=0)
        self.U, self.V, self.B = lfmm2(genotypes, environmental_factors, self.K, self.lambda_)
    def predict(self,
                target_environmental_factors: np.ndarray): # Environmental matrix (nxP)
        """
        Predicts the centered genotypes for a given environmental matrix.
        :param target_environmental_factors: 2D array of environmental factors (nxP)
        :return: 2D array of centered genotypes (nxL)
        """
        target_environmental_factors = target_environmental_factors - np.mean(target_environmental_factors, axis=0)
        return target_environmental_factors @ self.B.T + self.U@self.V.T

    def f_test(self,
               genotypes: np.ndarray,  # Allele frequency matrix (nxL)
               environmental_factors: np.ndarray,  # Environmental matrix (nxP)
               genomic_control: bool=True,  # whether to use genomic control
               ) -> tuple[ndarray, ndarray]: # Array of F-scores and p-values (L)
        """
        Make an F-test by re-fit the coefficient of the LFMM model.
        :param genotypes: 2D array of genotypes (nxL)
        :param environmental_factors: 2D array of environmental factors (nxP)
        :param genomic_control: Boolean, whether to use genomic control
        :return: A tuple of two arrays. First are the F-scores  and corresponding p-values
        """
        genotypes = genotypes - np.mean(genotypes, axis=0)
        environmental_factors = environmental_factors - np.mean(environmental_factors, axis=0)
        n, d = environmental_factors.shape
        u = np.hstack([np.ones((n, 1)), self.U])
        u_inv_y = np.linalg.lstsq(u, genotypes, rcond=None)[0]
        res_y = genotypes - u @ u_inv_y
        res_x = environmental_factors - u @ np.linalg.lstsq(u, environmental_factors, rcond=None)[0]
        res_x = np.hstack([np.ones((n, 1)), res_x])
        fitted = res_x @ np.linalg.lstsq(res_x, res_y, rcond=None)[0]
        mss = np.sum(np.abs(fitted) ** 2, axis=0)
        res_var = np.sum(np.abs(res_y - fitted) ** 2, axis=0) / (n - d - 1)
        #res_var = np.maximum(res_var, 1e-10)  # Numerical stability
        f_scores = mss / (d * res_var)
        dist = f(d, n - d - 1)
        if genomic_control:
            gif = np.median(f_scores) / dist.ppf(0.5)
            p_values = 1 - dist.cdf(f_scores / gif)
        else:
            p_values = 1 - dist.cdf(f_scores)
        return np.array(f_scores), np.array(p_values)