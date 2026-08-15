"""Reusable experiment helpers."""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from .systems import simulate
from .derivatives import estimate
from .sindy_models import build_model, fit_model


FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def add_relative_gaussian_noise(X, noise_level, rng):
    scale = np.std(X, axis=0, keepdims=True)
    return X + rng.normal(size=X.shape) * scale * noise_level


def derivative_experiment(system="vanderpol", noise_levels=(0, .01, .05, .10, .20)):
    t = np.linspace(0, 30, 3000)
    X = simulate(system, t)
    rng = np.random.default_rng(42)

    rows = []
    for noise in noise_levels:
        Xn = add_relative_gaussian_noise(X, noise, rng)
        true_dX = np.gradient(X, t, axis=0)

        for method in ["finite_difference", "smoothed_fd", "savgol"]:
            dX = estimate(Xn, t[1] - t[0], method)
            err = np.linalg.norm(dX - true_dX) / np.linalg.norm(true_dX)
            rows.append((noise, method, err))

    return rows


def plot_derivative_comparison(rows):
    methods = sorted(set(r[1] for r in rows))
    noises = sorted(set(r[0] for r in rows))
    plt.figure(figsize=(8, 5))
    for method in methods:
        y = [r[2] for r in rows if r[1] == method]
        plt.plot(noises, y, marker="o", label=method)
    plt.xlabel("Relative noise level")
    plt.ylabel("Relative derivative error")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "derivative_comparison.png", dpi=180)
    plt.close()
