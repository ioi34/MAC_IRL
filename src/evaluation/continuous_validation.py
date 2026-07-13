from __future__ import annotations

import numpy as np
import pandas as pd


CONTINUOUS_METRICS = ("direction_accuracy", "rmse", "correlation")


def aggregate_continuous_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby(
            ["observation_index", "date", "investor", "actual_action"],
            sort=True,
            as_index=False,
        )["predicted_action"]
        .mean()
    )


def continuous_metric_value(
    metric: str,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if metric == "direction_accuracy":
        return float((np.sign(actual) == np.sign(predicted)).mean())
    if metric == "rmse":
        return float(np.sqrt(np.square(predicted - actual).mean()))
    if metric == "correlation":
        if actual.std(ddof=0) == 0 or predicted.std(ddof=0) == 0:
            return np.nan
        return float(np.corrcoef(actual, predicted)[0, 1])
    raise ValueError(f"Unknown continuous metric: {metric}")


def continuous_degradation_delta(
    metric: str,
    actual: np.ndarray,
    baseline_predicted: np.ndarray,
    ablation_predicted: np.ndarray,
) -> float:
    baseline = continuous_metric_value(metric, actual, baseline_predicted)
    ablation = continuous_metric_value(metric, actual, ablation_predicted)
    if metric == "rmse":
        return ablation - baseline
    return baseline - ablation


def moving_block_bootstrap_continuous_deltas(
    metric: str,
    actual: np.ndarray,
    baseline_predicted: np.ndarray,
    ablation_predicted: np.ndarray,
    *,
    block_length: int = 20,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> np.ndarray:
    actual = np.asarray(actual, dtype=float)
    baseline_predicted = np.asarray(baseline_predicted, dtype=float)
    ablation_predicted = np.asarray(ablation_predicted, dtype=float)
    if not (len(actual) == len(baseline_predicted) == len(ablation_predicted)):
        raise ValueError("Actual, baseline, and ablation arrays must have equal length")
    if len(actual) == 0:
        raise ValueError("Cannot bootstrap an empty prediction set")
    if block_length < 1:
        raise ValueError("block_length must be positive")

    rng = np.random.default_rng(seed)
    blocks_per_sample = int(np.ceil(len(actual) / block_length))
    offsets = np.arange(block_length)
    deltas = np.empty(n_resamples, dtype=float)
    chunk_size = 500
    for chunk_start in range(0, n_resamples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_resamples)
        starts = rng.integers(
            0,
            len(actual),
            size=(chunk_end - chunk_start, blocks_per_sample),
        )
        indices = (
            (starts[:, :, None] + offsets[None, None, :]) % len(actual)
        ).reshape(chunk_end - chunk_start, -1)[:, : len(actual)]
        sampled_actual = actual[indices]
        sampled_baseline = baseline_predicted[indices]
        sampled_ablation = ablation_predicted[indices]
        if metric == "direction_accuracy":
            baseline_values = (
                np.sign(sampled_actual) == np.sign(sampled_baseline)
            ).mean(axis=1)
            ablation_values = (
                np.sign(sampled_actual) == np.sign(sampled_ablation)
            ).mean(axis=1)
            chunk_deltas = baseline_values - ablation_values
        elif metric == "rmse":
            baseline_values = np.sqrt(
                np.square(sampled_baseline - sampled_actual).mean(axis=1)
            )
            ablation_values = np.sqrt(
                np.square(sampled_ablation - sampled_actual).mean(axis=1)
            )
            chunk_deltas = ablation_values - baseline_values
        elif metric == "correlation":
            baseline_values = _row_correlation(sampled_actual, sampled_baseline)
            ablation_values = _row_correlation(sampled_actual, sampled_ablation)
            chunk_deltas = baseline_values - ablation_values
        else:
            raise ValueError(f"Unknown continuous metric: {metric}")
        deltas[chunk_start:chunk_end] = chunk_deltas
    return deltas


def _row_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=1)
    denominator = np.sqrt(
        np.sum(np.square(left_centered), axis=1)
        * np.sum(np.square(right_centered), axis=1)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )
