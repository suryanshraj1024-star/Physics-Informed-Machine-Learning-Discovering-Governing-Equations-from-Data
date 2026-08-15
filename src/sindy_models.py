"""PySINDy model construction and equation utilities."""
import numpy as np
import pysindy as ps


def build_model(
    library="poly3",
    threshold=0.1,
    differentiation_method="smoothed_fd",
):
    if library == "poly2":
        lib = ps.PolynomialLibrary(degree=2)
    elif library == "poly3":
        lib = ps.PolynomialLibrary(degree=3)
    elif library == "poly4":
        lib = ps.PolynomialLibrary(degree=4)
    elif library == "trig":
        lib = ps.PolynomialLibrary(degree=3)
        lib = lib + ps.FourierLibrary(n_frequencies=2)
    elif library == "poly_trig":
        lib = ps.PolynomialLibrary(degree=3) + ps.FourierLibrary(n_frequencies=2)
    else:
        raise ValueError(f"Unknown library: {library}")

    if differentiation_method == "finite_difference":
        diff = ps.FiniteDifference()
    else:
        diff = ps.SmoothedFiniteDifference()

    optimizer = ps.STLSQ(
        threshold=threshold,
        alpha=0.0,
        normalize_columns=True,
        max_iter=50,
    )
    return ps.SINDy(
        optimizer=optimizer,
        feature_library=lib,
        differentiation_method=diff,
    )


def fit_model(model, X, t, names):
    """Fit without leaking feature names into SINDy constructor."""
    model.fit(X, t=t, feature_names=names)
    return model


def coefficient_matrix(model):
    return np.asarray(model.coefficients())


def print_equations(model):
    model.print()
