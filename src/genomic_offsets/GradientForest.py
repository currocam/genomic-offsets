"""GradientForest"""

import numpy as np
from rpy2.robjects import default_converter
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import PackageNotInstalledError
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from numba import njit
from numba.typed import List


@njit
def _genomic_offset_helper(
    turnover, environmental_factors, target_environmental_factors
):
    current_f, future_f = np.zeros(environmental_factors.shape), np.zeros(
        environmental_factors.shape
    )
    for feature_idx in range(environmental_factors.shape[1]):
        current_f[:, feature_idx] = np.interp(
            environmental_factors[:, feature_idx],
            turnover[feature_idx][0],
            turnover[feature_idx][1],
        )
        future_f[:, feature_idx] = np.interp(
            target_environmental_factors[:, feature_idx],
            turnover[feature_idx][0],
            turnover[feature_idx][1],
        )
    return current_f - future_f


class GradientForestGO:
    """Gradient forest genomic offset statistic."""

    def __init__(
        self,
        n_trees: int,  # Number of  trees
    ):
        """
        Initializes the Gradient forest genomic offset model. It requires the gradientForest R package.
        :param n_trees: Number of trees.
        """
        self.n_trees = n_trees
        self._gf = None
        self.F = []
        self._predictors = None
        self._base = importr("base")
        try:
            # Try loading the gradientForest package
            self._gradient_forest = importr("gradientForest")
        except PackageNotInstalledError:
            msg = "gradientForest package is not installed. You can install it using:\n"
            msg += 'install.packages("gradientForest", repos="https://R-Forge.R-project.org")'
            raise PackageNotInstalledError(msg)
        except Exception as e:
            print(f"An error occurred: {e}")

    def __str__(self):
        return f"Gradient Forest genomic offset with {self.n_trees} trees."

    __repr__ = __str__

    def fit(
        self,
        allele_frequency: np.ndarray,  # Allele frequency matrix (nxL)
        environmental_factors: np.ndarray,
    ):  # Environmental matrix (nxP)
        """
        Fits the Geometric genomic offset model.
        :param allele_frequency: 2D array with allele frequencies.
        :param environmental_factors: 2D array with environmental factors.
        :return:
        """
        n1, l = allele_frequency.shape
        n2, p = environmental_factors.shape
        if n1 != n2:
            raise ValueError("Dimensions of array don't match")
        np_cv_rules = numpy2ri.converter + default_converter
        # Use the converter within this context
        with np_cv_rules.context():
            y_r = ro.r.matrix(
                np.asarray(allele_frequency), nrow=allele_frequency.shape[0]
            )
            x_r = ro.r.matrix(
                np.asarray(environmental_factors), nrow=environmental_factors.shape[0]
            )
            df = self._base.as_data_frame(ro.r.cbind(x_r, y_r))
            cols = self._base.colnames(df)
            predictor_vars = cols[: environmental_factors.shape[1]]
            self._gf = self._gradient_forest.gradientForest(
                df,
                predictor_vars=predictor_vars,
                response_vars=cols[environmental_factors.shape[1] :],
                ntree=self.n_trees,
            )
            self.F = List()
            for col in predictor_vars:
                dist = self._gradient_forest.cumimp_gradientForest(self._gf, col)
                dist = (np.array(dist[0]), np.array(dist[1]))
                self.F.append(dist)

    def importance_turnover(
        self,
        x: np.ndarray,  # Values of interest to interpolate
        feature_idx: int,  # Environmental covariate index
    ) -> np.ndarray:  # Interpolated importance turnover
        """
        Compute the importance turnover for the fitted model.
        :param x: Array with values where to compute the importance turnover.
        :param feature_idx: Index of the feature.
        :return: Array with importance turnover.
        """
        if self.F is None:
            raise ValueError("You have to fit the model first!")
        return np.interp(x, self.F[feature_idx][0], self.F[feature_idx][1])

    def genomic_offset(
        self,
        environmental_factors: np.ndarray,  # Environmental matrix (nxP)
        target_environmental_factors: np.ndarray,  # Altered environmental matrix (nxP)
    ) -> np.ndarray:  # A vector of genomic offsets (n)
        """
        Calculates the genomic offset statistic.
        :param environmental_factors: 2D array with environmental factors.
        :param target_environmental_factors: 2D array with target environmental factors.
        :return: Array with gradient forest genomic offset statistic.
        """
        if environmental_factors.shape != target_environmental_factors.shape:
            raise ValueError("Dimensions of array don't match")
        # Interpolate for each datapoint
        diff = _genomic_offset_helper(
            self.F, environmental_factors, target_environmental_factors
        )
        return np.linalg.norm(diff, ord=2, axis=1)
