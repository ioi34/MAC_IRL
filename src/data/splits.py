from __future__ import annotations

import numpy as np
from skfolio.model_selection import CombinatorialPurgedCV


def build_cpcv(config: dict) -> CombinatorialPurgedCV:
    cv = config["cross_validation"]
    return CombinatorialPurgedCV(
        n_folds=int(cv["n_folds"]),
        n_test_folds=int(cv["n_test_folds"]),
        purged_size=int(cv["purged_size"]),
        embargo_size=int(cv["embargo_size"]),
    )


def combine_test_folds(test_folds: list[np.ndarray] | np.ndarray) -> np.ndarray:
    if isinstance(test_folds, np.ndarray) and test_folds.ndim == 1:
        return test_folds.astype(int, copy=False)
    return np.sort(np.concatenate(list(test_folds))).astype(int, copy=False)
