from scripts.train import resolve_investor_config


def test_investor_override_keeps_training_configs_independent():
    config = {
        "epochs": 50,
        "learning_rate": 0.01,
        "loss": {"lambda_l1": 0.001},
        "investor_overrides": {
            "foreign": {
                "epochs": 100,
                "loss": {"lambda_l1": 0.0},
            }
        },
    }

    foreign = resolve_investor_config(config, "foreign")
    retail = resolve_investor_config(config, "retail")

    assert foreign["epochs"] == 100
    assert foreign["learning_rate"] == 0.01
    assert foreign["loss"]["lambda_l1"] == 0.0
    assert retail["epochs"] == 50
    assert retail["loss"]["lambda_l1"] == 0.001
