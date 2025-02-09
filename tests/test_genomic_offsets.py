import numpy as np
import src.genomic_offsets as go


def toy_dataset(rng, n, l, p, n_targets):
    x = rng.normal(size=n)
    b = np.zeros(l)
    target_indices = rng.choice(l, n_targets, replace=False)
    b[target_indices] = rng.uniform(-10, 10, size=n_targets)
    u = np.dot(x.reshape(-1, 1), np.array([[-1, 0.5, 1.5]])) + rng.normal(size=(n, 3))
    v = rng.normal(size=(3, l))  # v should have 3 rows to match the columns of u
    y = (
        np.dot(x.reshape(-1, 1), b.reshape(1, -1))
        + np.dot(u, v)
        + rng.normal(scale=0.5, size=(n, l))
    )
    y = (y > 0).astype(int)
    x = np.hstack((x.reshape(-1, 1), rng.normal(size=(n, p - 1))))
    assert x.shape == (n, p)
    assert y.shape == (n, l)
    return y, x


# In this suite of tests, we assert that some invariants hold for the genomic offsets
# implementations.


def test_rona():
    # simulate data
    rng = np.random.default_rng()
    n = rng.integers(low=1, high=300)
    l = rng.integers(low=1, high=1000)
    p = rng.integers(low=1, high=min(50, n - 1))
    n_targets = rng.integers(low=0, high=l)
    y_sim, x_sim = toy_dataset(rng, n=n, l=l, p=p, n_targets=n_targets)
    # simulate two diffrent environmental matrices
    n2 = rng.integers(low=1, high=300)
    x_test = rng.normal(size=(n2, p))
    x_test2 = rng.normal(size=(n2, p))
    # Fit model and make predictions
    model = go.RONA()
    model.fit(y_sim, x_sim)
    assert model.predict(x_sim).shape == y_sim.shape
    assert model.genomic_offset(x_test, x_test2).shape == n2
    assert model.genomic_offset(x_test, x_test).sum() == 0.0


def test_rda():
    # simulate data
    rng = np.random.default_rng()
    n = rng.integers(low=1, high=300)
    l = rng.integers(low=1, high=1000)
    p = rng.integers(low=1, high=min(50, n - 1))
    n_targets = rng.integers(low=0, high=l)
    y_sim, x_sim = toy_dataset(rng, n=n, l=l, p=p, n_targets=n_targets)
    # simulate two diffrent environmental matrices
    n2 = rng.integers(low=1, high=300)
    x_test = rng.normal(size=(n2, p))
    x_test2 = rng.normal(size=(n2, p))
    # Fit model and make predictions
    k = rng.integers(low=1, high=min(n, l))
    model = go.RDA(n_latent_factors=k)
    model.fit(y_sim, x_sim)
    assert model.predict(x_sim).shape == (n, k)
    assert model.genomic_offset(x_test, x_test2).shape == n2
    assert model.genomic_offset(x_test, x_test).sum() == 0.0
    # Fit model with covariates
    z = rng.integers(low=1, high=min(10, n - 1))
    covariates = rng.normal(size=(n, z))
    model = go.PartialRDA(n_latent_factors=k)
    model.fit(y_sim, x_sim, covariates)
    assert model.predict(x_sim, covariates).shape == (n, k)
    covariates2 = rng.normal(size=(n2, z))
    assert model.predict(x_test, covariates2).shape == (n2, k)
    assert model.genomic_offset(x_test, x_test2, covariates2).shape == n2


def test_geometric():
    # simulate data
    rng = np.random.default_rng()
    n = rng.integers(low=1, high=300)
    l = rng.integers(low=1, high=1000)
    p = rng.integers(low=1, high=min(50, n - 1))
    n_targets = rng.integers(low=0, high=l)
    y_sim, x_sim = toy_dataset(rng, n=n, l=l, p=p, n_targets=n_targets)
    # simulate two diffrent environmental matrices
    n2 = rng.integers(low=1, high=300)
    x_test = rng.normal(size=(n2, p))
    x_test2 = rng.normal(size=(n2, p))
    # Fit model and make predictions
    k = rng.integers(low=1, high=min(n, l))
    model = go.GeometricGO(n_latent_factors=k, lambda_=1e-5)
    model.fit(y_sim, x_sim)
    assert model.predict(x_sim).shape == (n, l)
    assert model.genomic_offset(x_test, x_test2).shape == n2
    assert model.genomic_offset(x_test, x_test).sum() == 0.0


def test_gradient_forest():
    rng = np.random.default_rng()
    n = rng.integers(low=1, high=300)
    # Less loci to decrease the computational cost
    l = rng.integers(low=1, high=200)
    p = rng.integers(low=1, high=min(50, n - 1))
    n_targets = rng.integers(low=0, high=l)
    y_sim, x_sim = toy_dataset(rng, n=n, l=l, p=p, n_targets=n_targets)
    # simulate two diffrent environmental matrices
    n2 = rng.integers(low=1, high=300)
    x_test = rng.normal(size=(n2, p))
    x_test2 = rng.normal(size=(n2, p))
    # Fit model and make predictions
    n_trees = rng.integers(low=10, high=500)
    try:
        model = go.GradientForestGO(n_trees=n_trees)
    except:
        return None
    model.fit(y_sim, x_sim)
    assert model.genomic_offset(x_test, x_test2).shape == n2
    assert model.genomic_offset(x_test, x_test).sum() == 0.0
