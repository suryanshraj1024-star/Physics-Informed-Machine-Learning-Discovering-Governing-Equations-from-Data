"""Train on one trajectory and validate the discovered law on a new IC."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.systems import simulate
from src.sindy_models import build_model, fit_model
from src.metrics import trajectory_metrics


def main():
    t_train = np.linspace(0, 12, 1200)
    t_test = np.linspace(0, 12, 1200)

    X_train = simulate("vanderpol", t_train, x0=[2.0, 0.0])
    X_test = simulate("vanderpol", t_test, x0=[-1.2, 1.0])

    model = build_model("poly3", threshold=0.05)
    fit_model(model, X_train, t_train, ["x", "y"])

    X_pred = model.simulate(X_test[0], t=t_test)
    print("Generalization metrics:", trajectory_metrics(X_test, X_pred))
    model.print()


if __name__ == "__main__":
    main()
