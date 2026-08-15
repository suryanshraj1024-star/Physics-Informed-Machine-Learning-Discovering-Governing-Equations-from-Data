"""Ground-truth dynamical systems used for controlled experiments."""
import numpy as np
from scipy.integrate import solve_ivp


def simulate(system, t, x0=None, **kwargs):
    """Simulate a named system with high-accuracy numerical integration."""
    if system == "vanderpol":
        mu = kwargs.get("mu", 1.5)
        x0 = np.asarray(x0 if x0 is not None else [2.0, 0.0], dtype=float)

        def rhs(ti, x):
            return [x[1], mu * (1.0 - x[0] ** 2) * x[1] - x[0]]

    elif system == "lorenz":
        sigma = kwargs.get("sigma", 10.0)
        rho = kwargs.get("rho", 28.0)
        beta = kwargs.get("beta", 8.0 / 3.0)
        x0 = np.asarray(x0 if x0 is not None else [1.0, 1.0, 1.0], dtype=float)

        def rhs(ti, x):
            return [
                sigma * (x[1] - x[0]),
                x[0] * (rho - x[2]) - x[1],
                x[0] * x[1] - beta * x[2],
            ]

    elif system == "pendulum":
        g = kwargs.get("g", 9.81)
        L = kwargs.get("L", 1.0)
        b = kwargs.get("b", 0.15)
        x0 = np.asarray(x0 if x0 is not None else [2.5, 0.0], dtype=float)

        def rhs(ti, x):
            return [x[1], -(g / L) * np.sin(x[0]) - b * x[1]]

    else:
        raise ValueError(f"Unknown system: {system}")

    sol = solve_ivp(
        rhs,
        (float(t[0]), float(t[-1])),
        x0,
        t_eval=t,
        rtol=1e-10,
        atol=1e-10,
        method="RK45",
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T


def true_equation_metadata(system):
    """Return ground-truth coefficient dictionaries for support/error metrics."""
    if system == "vanderpol":
        return [
            {"y": 1.0},
            {"x0": -1.0, "y0": 1.5, "x2 y": -1.5},
        ]
    if system == "lorenz":
        return [
            {"x0": -10.0, "x1": 10.0},
            {"x0": 28.0, "x1": -1.0, "x0 x2": -1.0},
            {"x2": -8.0 / 3.0, "x0 x1": 1.0},
        ]
    if system == "pendulum":
        return [
            {"x1": 1.0},
            {"sin(x0)": -9.81, "x1": -0.15},
        ]
    raise ValueError(system)
