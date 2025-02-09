import numpy as np
from src.genomic_offsets.RidgeLFMM import tracy_widom_statistics, RidgeLFMM
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter


def test_tracy_widom_statistics():
    # Suite cases against the AssocTest R package
    test_cases = [
        (
            np.array([5, 3, 1, 0]),
            np.array([-0.8242730, -0.6018554, -0.5552457, np.nan]),
        ),
        (
            np.array([35, 12, 10, 1]),
            np.array([-0.507243, -1.350166, -0.632759, np.nan]),
        ),
        (
            np.array([35, 12, 10, 1, 0]),
            np.array([-0.4632767, -1.2003935, -0.4407791, -0.5552457, np.nan]),
        ),
        (
            np.array([12, 35, 10, 1, 0]),
            np.array([-0.4632767, -1.2003935, -0.4407791, -0.5552457, np.nan]),
        ),
    ]

    for eigenvalues, expected in test_cases:
        assert np.nansum(np.abs(tracy_widom_statistics(eigenvalues) - expected)) < 1e-5


def generative_model(rng, N, L, P, n_targets):
    x = rng.normal(size=N)
    b = np.zeros(L)
    target_indices = rng.choice(L, n_targets, replace=False)
    b[target_indices] = rng.uniform(-10, 10, size=n_targets)
    u = np.dot(x.reshape(-1, 1), np.array([[-1, 0.5, 1.5]])) + rng.normal(size=(N, 3))
    v = rng.normal(size=(3, L))  # v should have 3 rows to match the columns of u
    y = (
        np.dot(x.reshape(-1, 1), b.reshape(1, -1))
        + np.dot(u, v)
        + rng.normal(scale=0.5, size=(N, L))
    )
    y = (y > 0).astype(int)
    x = np.hstack((x.reshape(-1, 1), rng.normal(size=(N, P - 1))))
    assert x.shape == (N, P)
    assert y.shape == (N, L)
    return y, x


def test_lfmm():
    try:
        LEA = importr("LEA")
    except ImportError:
        # Install
        # Skip if not installed
        return

    # Create a converter that starts with rpy2's default converter
    # to which the numpy conversion rules are added.
    np_cv_rules = default_converter + numpy2ri.converter

    with np_cv_rules.context():
        # Anything here and until the `with` block is exited
        # will use our numpy converter whenever objects are
        # passed to R or are returned by R while calling
        # rpy2.robjects functions.
        rng = np.random.default_rng()
        n = rng.integers(low=1, high=300)
        l = rng.integers(low=1, high=1000)
        p = rng.integers(low=1, high=min(50, n - 1))
        n_targets = rng.integers(low=0, high=l)
        # Simulate data
        y_sim, x_sim = generative_model(rng, N=n, L=l, P=p, n_targets=n_targets)
        # Simulate hyperparameters
        Ks = rng.integers(low=1, high=min(2, min(l, p)), size=5)
        # Generate random regularization parameters when fixed version of LEA
        # is available
        # See https://github.com/bcm-uga/LEA/commit/a37ac6c3b25b3d1b0ca0fa7517dcfaaf340fa631
        # lambdas = rng.uniform(low=1e-6, high=1e3, size=5)
        lambdas = np.repeat(1e-10, 5)
        for K, lambda_ in zip(Ks, lambdas):
            modelsim = RidgeLFMM(K, lambda_)
            modelsim.fit(y_sim, x_sim)
            r_model = LEA.lfmm2(y_sim, x_sim, modelsim.K, modelsim.lambda_, True)
            assert np.linalg.norm(np.asarray(r_model.slots["U"]) - modelsim.U) < 1e-5
            assert np.linalg.norm(np.asarray(r_model.slots["B"]) - modelsim.B) < 1e-5
            assert np.linalg.norm(np.asarray(r_model.slots["V"]) - modelsim.V) < 1e-5
            # Test F-test
            use_genomic_control = bool(rng.choice([True, False]))
            r_test = LEA.lfmm2_test(r_model, y_sim, x_sim, True, use_genomic_control)
            fscores, p_values = modelsim.f_test(y_sim, x_sim, use_genomic_control)
            assert np.linalg.norm(r_test["fscores"] - fscores) < 1e-5
            assert np.linalg.norm(r_test["pvalues"] - p_values) < 1e-5
