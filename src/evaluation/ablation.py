from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def remove_features(config: dict, features_to_remove: set[str]) -> dict:
    updated = deepcopy(config)
    updated["features"]["selected"] = [
        name for name in updated["features"]["selected"] if name not in features_to_remove
    ]
    return updated


def build_ablation_variants(
    feature_names: list[str],
    groups: dict[str, set[str]] | None = None,
) -> dict[str, set[str]]:
    variants = {f"remove_{name}": {name} for name in feature_names}
    for name, removed in (groups or {}).items():
        unknown = removed.difference(feature_names)
        if unknown:
            raise ValueError(f"Ablation group '{name}' contains unknown features: {sorted(unknown)}")
        variants[name] = set(removed)
    return variants


def aggregate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    probability_columns = sorted(
        (column for column in predictions if column.startswith("probability_")),
        key=lambda column: int(column.rsplit("_", 1)[1]),
    )
    grouped = (
        predictions.groupby(
            ["observation_index", "date", "investor", "y_true"],
            sort=True,
            as_index=False,
        )[probability_columns]
        .mean()
    )
    probability_sum = grouped[probability_columns].sum(axis=1)
    grouped[probability_columns] = grouped[probability_columns].div(probability_sum, axis=0)
    grouped["y_pred"] = grouped[probability_columns].to_numpy().argmax(axis=1)
    return grouped


def metric_value(
    metric: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    predicted = probabilities.argmax(axis=1)
    labels = list(range(probabilities.shape[1]))
    if metric == "accuracy":
        return float(accuracy_score(y_true, predicted))
    if metric == "macro_f1":
        return float(
            f1_score(y_true, predicted, average="macro", labels=labels, zero_division=0)
        )
    if metric == "nll":
        normalized = probabilities / probabilities.sum(axis=1, keepdims=True)
        selected = normalized[np.arange(len(y_true)), y_true]
        return float(-np.log(np.clip(selected, 1e-15, 1.0)).mean())
    raise ValueError(f"Unknown metric: {metric}")


def degradation_delta(
    metric: str,
    y_true: np.ndarray,
    baseline_probabilities: np.ndarray,
    ablation_probabilities: np.ndarray,
) -> float:
    baseline = metric_value(metric, y_true, baseline_probabilities)
    ablation = metric_value(metric, y_true, ablation_probabilities)
    return ablation - baseline if metric == "nll" else baseline - ablation


def moving_block_bootstrap_deltas(
    metric: str,
    y_true: np.ndarray,
    baseline_probabilities: np.ndarray,
    ablation_probabilities: np.ndarray,
    *,
    block_length: int = 20,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> np.ndarray:
    n_observations = len(y_true)
    if n_observations == 0:
        raise ValueError("Cannot bootstrap an empty prediction set")
    if block_length < 1:
        raise ValueError("block_length must be positive")
    baseline_probabilities = baseline_probabilities / baseline_probabilities.sum(
        axis=1, keepdims=True
    )
    ablation_probabilities = ablation_probabilities / ablation_probabilities.sum(
        axis=1, keepdims=True
    )
    rng = np.random.default_rng(seed)
    blocks_per_sample = int(np.ceil(n_observations / block_length))
    offsets = np.arange(block_length)
    deltas = np.empty(n_resamples, dtype=float)
    chunk_size = 500
    baseline_predicted = baseline_probabilities.argmax(axis=1)
    ablation_predicted = ablation_probabilities.argmax(axis=1)
    baseline_losses = -np.log(
        np.clip(baseline_probabilities[np.arange(n_observations), y_true], 1e-15, 1.0)
    )
    ablation_losses = -np.log(
        np.clip(ablation_probabilities[np.arange(n_observations), y_true], 1e-15, 1.0)
    )
    for chunk_start in range(0, n_resamples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_resamples)
        starts = rng.integers(
            0,
            n_observations,
            size=(chunk_end - chunk_start, blocks_per_sample),
        )
        indices = (
            (starts[:, :, None] + offsets[None, None, :]) % n_observations
        ).reshape(chunk_end - chunk_start, -1)[:, :n_observations]
        sampled_true = y_true[indices]
        if metric == "accuracy":
            baseline_values = (baseline_predicted[indices] == sampled_true).mean(axis=1)
            ablation_values = (ablation_predicted[indices] == sampled_true).mean(axis=1)
            chunk_deltas = baseline_values - ablation_values
        elif metric == "nll":
            chunk_deltas = (
                ablation_losses[indices].mean(axis=1)
                - baseline_losses[indices].mean(axis=1)
            )
        elif metric == "macro_f1":
            baseline_values = _binary_macro_f1_rows(
                sampled_true, baseline_predicted[indices]
            )
            ablation_values = _binary_macro_f1_rows(
                sampled_true, ablation_predicted[indices]
            )
            chunk_deltas = baseline_values - ablation_values
        else:
            raise ValueError(f"Unknown metric: {metric}")
        deltas[chunk_start:chunk_end] = chunk_deltas
    return deltas


def _binary_macro_f1_rows(y_true: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    scores = []
    for label in (0, 1):
        true_label = y_true == label
        predicted_label = predicted == label
        true_positive = (true_label & predicted_label).sum(axis=1)
        false_positive = (~true_label & predicted_label).sum(axis=1)
        false_negative = (true_label & ~predicted_label).sum(axis=1)
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(
            np.divide(
                2 * true_positive,
                denominator,
                out=np.zeros_like(denominator, dtype=float),
                where=denominator != 0,
            )
        )
    return np.mean(scores, axis=0)


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return pd.Series(result, index=p_values.index)


def feature_vif(features: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    if features.ndim != 2 or features.shape[1] != len(feature_names):
        raise ValueError("features must be a 2D matrix matching feature_names")
    rows = []
    for feature_idx, feature in enumerate(feature_names):
        target = features[:, feature_idx]
        others = np.delete(features, feature_idx, axis=1)
        design = np.column_stack([np.ones(len(others)), others])
        fitted = design @ np.linalg.lstsq(design, target, rcond=None)[0]
        residual_sum = float(np.square(target - fitted).sum())
        total_sum = float(np.square(target - target.mean()).sum())
        if total_sum == 0:
            vif = np.inf
        else:
            r_squared = 1.0 - residual_sum / total_sum
            vif = np.inf if r_squared >= 1.0 - 1e-12 else 1.0 / (1.0 - r_squared)
        rows.append({"feature": feature, "vif": vif, "high_vif": bool(vif >= 5.0)})
    return pd.DataFrame(rows)
