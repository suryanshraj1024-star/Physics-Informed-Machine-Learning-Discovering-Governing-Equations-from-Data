"""Derivative estimation methods for noisy observations."""
import numpy as np
from scipy.signal import savgol_filter


def finite_difference(X, dt):
    X = np.asarray(X)
    return np.gradient(X, dt, axis=0, edge_order=2)


def smoothed_finite_difference(X, dt, window=21, polyorder=3):
    X = np.asarray(X)
    window = min(window, len(X) - (1 - len(X) % 2))
    if window % 2 == 0:
        window -= 1
    window = max(window, polyorder + 2 + (polyorder + 2) % 2)
    return np.gradient(
        savgol_filter(X, window_length=window, polyorder=polyorder, axis=0),
        dt,
        axis=0,
        edge_order=2,
    )


def savgol_derivative(X, dt, window=21, polyorder=3):
    X = np.asarray(X)
    window = min(window, len(X) - (1 - len(X) % 2))
    if window % 2 == 0:
        window -= 1
    window = max(window, polyorder + 2 + (polyorder + 2) % 2)
    return savgol_filter(
        X, window_length=window, polyorder=polyorder,
        deriv=1, delta=dt, axis=0
    )


def estimate(X, dt, method="smoothed_fd"):
    if method == "finite_difference":
        return finite_difference(X, dt)
    if method == "smoothed_fd":
        return smoothed_finite_difference(X, dt)
    if method == "savgol":
        return savgol_derivative(X, dt)
    raise ValueError(f"Unknown derivative method: {method}")
