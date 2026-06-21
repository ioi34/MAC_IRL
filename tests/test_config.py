from src.utils.config import load_configs


def test_default_feature_config_matches_registry_schema():
    config = load_configs("configs/data.yaml", "configs/features.yaml")

    for feature in config["features"]["selected"]:
        assert feature in config["features"]["params"]
