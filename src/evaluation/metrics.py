"""Evaluation metrics."""
import numpy as np


def mape(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(y_true, 1)) * 100)


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE — better for zero-heavy data."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(denom, 1e-8)) * 100)


def pinball_loss(y_true, y_pred, quantile: float) -> float:
    """Quantile loss for probabilistic forecasts."""
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))
