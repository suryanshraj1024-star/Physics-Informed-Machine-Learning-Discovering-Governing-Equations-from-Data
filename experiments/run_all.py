"""Run the core controlled SINDy study."""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.systems import simulate
from src.sindy_models import build_model, fit_model


def main():
    out = Path("results/figures")
    out.mkdir(parents=True, exist_ok=True)

    # 1. Van der Pol: clean equation recovery
    t = np.linspace(0, 30, 3000)
    X = simulate("vanderpol", t)
    model = build_model("poly3", threshold=0.05)
    fit_model(model, X, t, ["x", "y"])

    print("\n=== Van der Pol ===")
    model.print()

    # 2. Lorenz: exact quadratic structure
    t_l = np.arange(0, 20, 0.002)
    X_l = simulate("lorenz", t_l)
    lorenz = build_model("poly2", threshold=0.05)
    fit_model(lorenz, X_l, t_l, ["x", "y", "z"])

    print("\n=== Lorenz ===")
    lorenz.print()

    # 3. Pendulum: trig library
    t_p = np.linspace(0, 20, 3000)
    X_p = simulate("pendulum", t_p)
    pend = build_model("poly_trig", threshold=0.05)
    fit_model(pend, X_p, t_p, ["theta", "omega"])

    print("\n=== Pendulum ===")
    pend.print()

    # Basic summary plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(X[:, 0], X[:, 1])
    axes[0].set_title("Van der Pol")
    axes[1].plot(X_l[:, 0], X_l[:, 2])
    axes[1].set_title("Lorenz")
    axes[2].plot(X_p[:, 0], X_p[:, 1])
    axes[2].set_title("Damped Pendulum")
    for ax in axes:
        ax.set_xlabel("state 1")
        ax.set_ylabel("state 2")
    fig.tight_layout()
    fig.savefig(out / "summary_figure.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
