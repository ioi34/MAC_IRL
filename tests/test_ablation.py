import numpy as np
import pandas as pd
import pytest

from src.evaluation.ablation import (
    aggregate_predictions,
    benjamini_hochberg,
    build_ablation_variants,
    feature_vif,
    moving_block_bootstrap_deltas,
    remove_features,
)


def test_dynamic_ablation_variants_remove_each_selected_feature_and_preserve_source():
    source = {"features": {"selected": ["underwater", "momentum", "relative"]}}
    variants = build_ablation_variants(
        source["features"]["selected"],
        {"remove_traditional_group": {"momentum", "relative"}},
    )

    assert variants["remove_underwater"] == {"underwater"}
    assert variants["remove_traditional_group"] == {"momentum", "relative"}
    updated = remove_features(source, variants["remove_momentum"])
    assert updated["features"]["selected"] == ["underwater", "relative"]
    assert source["features"]["selected"] == ["underwater", "momentum", "relative"]


def test_aggregate_predictions_averages_repeated_cpcv_probabilities():
    predictions = pd.DataFrame(
        {
            "split": [0, 1],
            "observation_index": [3, 3],
            "date": ["2025-01-02", "2025-01-02"],
            "investor": ["foreign", "foreign"],
            "y_true": [1, 1],
            "probability_0": [0.4, 0.2],
            "probability_1": [0.6, 0.8],
        }
    )

    aggregated = aggregate_predictions(predictions)

    assert aggregated.loc[0, "probability_0"] == pytest.approx(0.3)
    assert aggregated.loc[0, "probability_1"] == pytest.approx(0.7)
    assert aggregated.loc[0, "y_pred"] == 1


def test_moving_block_bootstrap_is_reproducible():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    baseline = np.array([[0.8, 0.2], [0.2, 0.8]] * 3)
    ablation = np.full((6, 2), 0.5)

    first = moving_block_bootstrap_deltas(
        "nll", y_true, baseline, ablation, block_length=2, n_resamples=20, seed=7
    )
    second = moving_block_bootstrap_deltas(
        "nll", y_true, baseline, ablation, block_length=2, n_resamples=20, seed=7
    )

    np.testing.assert_allclose(first, second)
    assert (first > 0).all()


def test_vif_flags_collinear_features_and_bh_is_monotone():
    x = np.arange(1, 21, dtype=float)
    features = np.column_stack([x, x * 2, np.sin(x)])
    vif = feature_vif(features, ["a", "b", "c"])

    assert np.isinf(vif.loc[vif["feature"] == "a", "vif"]).item()
    assert vif.loc[vif["feature"] == "a", "high_vif"].item()
    adjusted = benjamini_hochberg(pd.Series([0.01, 0.04, 0.03]))
    assert adjusted.tolist() == pytest.approx([0.03, 0.04, 0.04])
