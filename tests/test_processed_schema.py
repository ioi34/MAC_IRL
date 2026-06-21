import numpy as np
import pytest

from scripts.train import validate_processed_schema


def _config() -> dict:
    return {
        "features": {"selected": ["underwater", "herd"]},
        "investors": ["foreign", "institution", "retail"],
        "action_values": [-1, 1],
    }


def test_processed_schema_rejects_stale_dataset_without_feature_names():
    data = {
        "features": np.empty((2, 3, 2, 2)),
        "labels": np.empty((2, 3)),
        "dates": np.array(["2025-01-01", "2025-01-02"]),
        "action_values": np.array([-1, 1]),
    }

    with pytest.raises(ValueError, match="stale or incomplete"):
        validate_processed_schema(data, _config())


def test_processed_schema_rejects_different_feature_meanings():
    data = {
        "features": np.empty((2, 3, 2, 2)),
        "labels": np.empty((2, 3)),
        "dates": np.array(["2025-01-01", "2025-01-02"]),
        "action_values": np.array([-1, 1]),
        "feature_names": np.array(["loss_aversion", "herd"]),
        "investors": np.array(["foreign", "institution", "retail"]),
    }

    with pytest.raises(ValueError, match="do not match config"):
        validate_processed_schema(data, _config())
