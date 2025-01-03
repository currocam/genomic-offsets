"""GradientForest"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/05_gf.ipynb.

# %% auto 0
__all__ = ['GradientForestGO']

# %% ../nbs/05_gf.ipynb 3
from fastcore.utils import *
import statsmodels.api as sm
from nbdev.showdoc import *
from scipy.linalg import svd, inv
from scipy.stats import f
from TracyWidom import TracyWidom
from numba import njit, jit
from collections import namedtuple
# Rpy2
from rpy2.robjects.packages import importr
from rpy2.robjects import default_converter
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import PackageNotInstalledError
import rpy2.robjects as ro
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% ../nbs/05_gf.ipynb 6
class GradientForestGO:
    "Gradient forest genomic offset statistic."
    def __init__(self, 
                 ntrees: int, # Number of  trees
                ):
        self.ntrees = ntrees
        self._gf = None
        self.F = []
        self._predictors = None
        self._base = importr('base')
        try:
            # Try loading the gradientForest package
            self._gradient_forest = importr('gradientForest')
        except PackageNotInstalledError:
            print("gradientForest package is not installed. You can install it using:")
            print('install.packages("gradientForest", repos="http://R-Forge.R-project.org")')
        except Exception as e:
            print(f"An error occurred: {e}")
    def __str__(self):
        return f"Gradient Forest genomic offset with {self.ntrees} trees."
    __repr__ = __str__

# %% ../nbs/05_gf.ipynb 10
@patch
def fit(self:GradientForestGO,
        Y: np.ndarray, # Allele frequency matrix (nxL)
        X: np.ndarray): # Environmental matrix (nxP)
    "Fits the Geometric genomic offset model. "
    n1, L = Y.shape
    n2, P = X.shape
    if n1 != n2: 
        raise ValueError("Dimensions of array don't match")
    np_cv_rules = numpy2ri.converter+default_converter
    # Use the converter within this context
    with np_cv_rules.context():
        Y_r = ro.r.matrix(np.asarray(Y), nrow=Y.shape[0])
        X_r = ro.r.matrix(np.asarray(X), nrow=X.shape[0])
        df = self._base.as_data_frame(ro.r.cbind(X_r, Y_r))
        cols = self._base.colnames(df)
        predictor_vars = cols[:X.shape[1]]
        self._gf = self._gradient_forest.gradientForest(
            df, predictor_vars=predictor_vars,
            response_vars=cols[X.shape[1]:], ntree=500
        )
        self.F = []
        for col in predictor_vars:
            dist = self._gradient_forest.cumimp_gradientForest(self._gf, col)
            dist = (np.array(dist[0]), np.array(dist[1]))
            self.F.append(dist)

# %% ../nbs/05_gf.ipynb 20
@patch
def importance_turnover(self:GradientForestGO,
        x: np.ndarray, # Values of interest to interpolate
        feature_idx: int, # Environmental covariate index
        )-> np.ndarray: # Interpolated importance turnover
    if self.F is None: 
        raise ValueError("You have to fit the model first!")
    return np.interp(x, self.F[feature_idx][0], self.F[feature_idx][1])

# %% ../nbs/05_gf.ipynb 25
@patch
def genomic_offset(self:GradientForestGO,
        X: np.ndarray, # Environmental matrix (nxP)
        Xstar: np.ndarray, # Altered environmental matrix (nxP)
           )-> np.ndarray: # A vector of genomic offsets (n)
    "Calculates the genomic offset statistic. " 
    if X.shape != Xstar.shape: 
        raise ValueError("Dimensions of array don't match")
    # Interpolate for each datapoint
    diff = _genomic_offset_helper(self.F, X, Xstar)
    return np.linalg.norm(diff, ord=2, axis=1)

