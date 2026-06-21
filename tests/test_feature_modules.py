from src.features.registry import FEATURE_REGISTRY


def test_each_reward_feature_is_owned_by_its_own_module():
    expected_modules = {
        "loss_aversion": "src.features.loss_aversion",
        "herd": "src.features.herd",
        "momentum": "src.features.momentum",
        "volatility": "src.features.volatility",
    }

    assert {name: builder.__module__ for name, builder in FEATURE_REGISTRY.items()} == expected_modules
