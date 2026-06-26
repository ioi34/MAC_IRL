from src.utils.config import load_configs


def test_default_feature_config_matches_registry_schema():
    config = load_configs("configs/data_extended.yaml", "configs/features_underwater.yaml")

    for feature in config["features"]["selected"]:
        assert feature in config["features"]["params"]
