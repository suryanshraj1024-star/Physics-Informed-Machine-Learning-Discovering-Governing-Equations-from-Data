"""Compare candidate libraries on Van der Pol."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.systems import simulate
from src.sindy_models import build_model, fit_model
from src.metrics import trajectory_metrics


def main():
    t = np.linspace(0, 30, 3000)
    X = simulate("vanderpol", t)

    for library in ["poly2", "poly3", "poly4", "trig", "poly_trig"]:
        model = build_model(library, threshold=0.05)
        fit_model(model, X, t, ["x", "y"])
        pred = model.simulate(X[0], t=t)
        metrics = trajectory_metrics(X, pred)
        active = int(np.count_nonzero(np.abs(model.coefficients()) > 1e-8))
        print(
            f"{library:10s} | R2={metrics['r2']:.5f} | "
            f"RMSE={metrics['rmse']:.5e} | active_terms={active}"
        )


if __name__ == "__main__":
    main()
