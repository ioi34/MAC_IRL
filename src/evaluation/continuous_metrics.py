from __future__ import annotations

import numpy as np


def evaluate_continuous_actions(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    finite = np.isfinite(actual) & np.isfinite(predicted)
    if not finite.all():
        actual = actual[finite]
        predicted = predicted[finite]
        error = error[finite]
    if len(actual) == 0:
        raise ValueError("No finite observations available for continuous-action metrics")

    actual_std = float(actual.std(ddof=0))
    predicted_std = float(predicted.std(ddof=0))
    correlation = np.nan
    if actual_std > 0 and predicted_std > 0:
        correlation = float(np.corrcoef(actual, predicted)[0, 1])

    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "mse": float(np.square(error).mean()),
        "direction_accuracy": float((np.sign(actual) == np.sign(predicted)).mean()),
        "correlation": correlation,
        "saturation_rate": float((np.abs(predicted) >= 1.0 - 1e-6).mean()),
        "actual_mean": float(actual.mean()),
        "predicted_mean": float(predicted.mean()),
    }
