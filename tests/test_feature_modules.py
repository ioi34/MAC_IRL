from src.features.registry import FEATURE_REGISTRY


def test_each_reward_feature_is_owned_by_its_own_module():
    expected_modules = {
        "underwater": "src.features.underwater",
        "herd": "src.features.herd",
        "momentum": "src.features.momentum",
        "volatility": "src.features.volatility",
        "persist": "src.features.persist",
        "turnover_20": "src.features.turnover",
        "turnover_60": "src.features.turnover",
        "turnover_120": "src.features.turnover",
        "herd_from_foreign": "src.features.herd_pairwise",
        "herd_from_institution": "src.features.herd_pairwise",
        "herd_from_retail": "src.features.herd_pairwise",
    }

    assert {name: builder.__module__ for name, builder in FEATURE_REGISTRY.items()} == expected_modules
