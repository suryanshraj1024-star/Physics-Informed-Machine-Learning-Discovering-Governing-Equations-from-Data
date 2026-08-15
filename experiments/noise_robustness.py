"""Repeated noise experiment: quantify stability rather than relying on one seed."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.systems import simulate
from src.experiments import add_relative_gaussian_noise
from src.sindy_models import build_model, fit_model


def main():
    t = np.linspace(0, 30, 3000)
    clean = simulate("vanderpol", t)
    rng = np.random.default_rng(7)

    for noise in [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]:
        recovered = []
        for _ in range(30):
            Xn = add_relative_gaussian_noise(clean, noise, rng)
            model = build_model("poly3", threshold=0.1)
            fit_model(model, Xn, t, ["x", "y"])
            recovered.append(np.linalg.norm(model.coefficients()))
        print(
            f"noise={noise:0.2f} | "
            f"coefficient-norm mean={np.mean(recovered):.4f} "
            f"std={np.std(recovered):.4f}"
        )


if __name__ == "__main__":
    main()
