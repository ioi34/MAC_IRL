from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


def fit_feature_scaler(features: np.ndarray, train_indices: np.ndarray) -> StandardScaler:
    train = features[train_indices]
    return StandardScaler().fit(train.reshape(-1, train.shape[-1]))


def fit_context_scaler(contexts: np.ndarray, train_indices: np.ndarray) -> StandardScaler:
    return StandardScaler().fit(contexts[train_indices])


def transform_context_matrix(contexts: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(contexts).astype(np.float32)


def transform_feature_tensor(features: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    shape = features.shape
    transformed = scaler.transform(features.reshape(-1, shape[-1]))
    return transformed.reshape(shape).astype(np.float32)


def save_feature_scaler(scaler: StandardScaler, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, target)


def load_feature_scaler(path: str | Path) -> StandardScaler:
    return joblib.load(path)
