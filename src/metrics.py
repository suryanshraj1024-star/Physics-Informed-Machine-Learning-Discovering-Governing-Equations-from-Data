"""Metrics for equation discovery and trajectory validation."""
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error


def coefficient_relative_error(Xi_hat, Xi_true):
    denom = np.linalg.norm(Xi_true)
    return np.linalg.norm(Xi_hat - Xi_true) / max(denom, 1e-15)


def support_scores(Xi_hat, Xi_true, tol=1e-8):
    pred = np.abs(Xi_hat) > tol
    true = np.abs(Xi_true) > tol
    tp = np.logical_and(pred, true).sum()
    fp = np.logical_and(pred, ~true).sum()
    fn = np.logical_and(~pred, true).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-15)
    return precision, recall, f1


def trajectory_metrics(X_true, X_pred):
    X_true = np.asarray(X_true)
    X_pred = np.asarray(X_pred)
    rmse = np.sqrt(mean_squared_error(X_true, X_pred))
    r2 = r2_score(X_true, X_pred, multioutput="uniform_average")
    return {"rmse": float(rmse), "r2": float(r2)}
