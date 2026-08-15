import numpy as np
from src.systems import simulate


def test_vanderpol_shape():
    t = np.linspace(0, 2, 100)
    X = simulate("vanderpol", t)
    assert X.shape == (100, 2)
    assert np.all(np.isfinite(X))


def test_lorenz_shape():
    t = np.linspace(0, 1, 100)
    X = simulate("lorenz", t)
    assert X.shape == (100, 3)
    assert np.all(np.isfinite(X))


def test_pendulum_shape():
    t = np.linspace(0, 2, 100)
    X = simulate("pendulum", t)
    assert X.shape == (100, 2)
    assert np.all(np.isfinite(X))
