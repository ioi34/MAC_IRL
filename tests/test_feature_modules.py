from src.features.registry import FEATURE_REGISTRY


def test_each_reward_feature_is_owned_by_its_own_module():
    expected_modules = {
        "underwater": "src.features.underwater",
        "herd": "src.features.herd",
        "herd_a": "src.features.herd",
        "herd_b": "src.features.herd",
        "execution_persistence_3": "src.features.institution",
        "short_residual_return_1": "src.features.institution",
        "short_residual_return_5": "src.features.institution",
        "benchmark_drift_20": "src.features.institution",
        "momentum": "src.features.momentum",
        "shortmom_orth": "src.features.momentum",
        "relative": "src.features.relative",
        "usd_momentum": "src.features.usd_momentum",
        "volatility": "src.features.volatility",
        "fx_sensitivity_1d": "src.features.fx",
        "turnover_20": "src.features.turnover",
        "turnover_60": "src.features.turnover",
        "turnover_120": "src.features.turnover",
    }

    assert {name: builder.__module__ for name, builder in FEATURE_REGISTRY.items()} == expected_modules
