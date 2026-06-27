import numpy as np
import pytest

from scripts.train import validate_processed_schema


def _config() -> dict:
    return {
        "features": {"selected": ["underwater", "herd"]},
        "investors": ["foreign", "institution", "retail"],
        "action_values": [-2, -1, 1, 2],
    }


def test_processed_schema_rejects_stale_dataset_without_feature_names():
    data = {
        "features": np.empty((2, 3, 4, 2)),
        "labels": np.empty((2, 3)),
        "dates": np.array(["2025-01-01", "2025-01-02"]),
        "action_values": np.array([-2, -1, 1, 2]),
    }

    with pytest.raises(ValueError, match="stale or incomplete"):
        validate_processed_schema(data, _config())


def test_processed_schema_rejects_different_feature_meanings():
    data = {
        "features": np.empty((2, 3, 4, 2)),
        "labels": np.empty((2, 3)),
        "dates": np.array(["2025-01-01", "2025-01-02"]),
        "action_values": np.array([-2, -1, 1, 2]),
        "feature_names": np.array(["loss_aversion", "herd"]),
        "investors": np.array(["foreign", "institution", "retail"]),
    }

    with pytest.raises(ValueError, match="do not match config"):
        validate_processed_schema(data, _config())


def test_contextual_schema_requires_context_arrays():
    config = _config()
    config["model"] = {"context_names": ["fx_return_1d"]}
    data = {
        "features": np.empty((2, 3, 4, 2)),
        "labels": np.empty((2, 3)),
        "dates": np.array(["2025-01-01", "2025-01-02"]),
        "action_values": np.array([-2, -1, 1, 2]),
        "feature_names": np.array(["underwater", "herd"]),
        "investors": np.array(["foreign", "institution", "retail"]),
    }

    with pytest.raises(ValueError, match="missing contextual arrays"):
        validate_processed_schema(data, config)


def test_contextual_schema_rejects_unknown_selected_context():
    config = _config()
    config["model"] = {"context_names": ["fx_return_5d"]}
    data = {
        "features": np.empty((2, 3, 4, 2)),
        "contexts": np.empty((2, 1)),
        "labels": np.empty((2, 3)),
        "dates": np.array(["2025-01-01", "2025-01-02"]),
        "action_values": np.array([-2, -1, 1, 2]),
        "feature_names": np.array(["underwater", "herd"]),
        "context_names": np.array(["fx_return_1d"]),
        "investors": np.array(["foreign", "institution", "retail"]),
    }

    with pytest.raises(ValueError, match="absent from processed contexts"):
        validate_processed_schema(data, config)
