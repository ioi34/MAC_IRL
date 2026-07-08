import numpy as np
import pandas as pd
import torch

from src.data.continuous_labels import add_continuous_action_labels, continuous_actions_array
from src.features.continuous import build_state_feature_tensor, valid_rows_for_continuous_model
from src.models.continuous import ContinuousInvestorIRLModel


def test_continuous_actions_are_trade_imbalance_shifted_to_next_day():
    frame = pd.DataFrame(
        {
            "foreign_buy_value": [3.0, 1.0, 4.0],
            "foreign_sell_value": [1.0, 3.0, 0.0],
        }
    )
    config = {
        "investors": ["foreign"],
        "columns": {
            "investor_trades": {
                "foreign": {
                    "buy_value": "foreign_buy_value",
                    "sell_value": "foreign_sell_value",
                }
            }
        },
        "labeling": {"label_shift": 1},
    }

    labeled = add_continuous_action_labels(frame, config)

    np.testing.assert_allclose(labeled["continuous_action_foreign"].iloc[:2], [-0.5, 1.0])
    assert pd.isna(labeled["continuous_action_foreign"].iloc[2])
    np.testing.assert_allclose(continuous_actions_array(labeled.iloc[:2], ["foreign"]), [[-0.5], [1.0]])


def test_continuous_state_features_use_action_one_without_action_axis():
    frame = pd.DataFrame(
        {
            "u_foreign": [0.1, 0.2, 0.3, 0.4],
            "u_institution": [0.0, -0.1, -0.2, -0.3],
            "u_retail": [-0.1, -0.2, -0.3, -0.4],
            "price": [100.0, 101.0, 103.0, 102.0],
            "continuous_action_foreign": [0.2, -0.1, 0.4, -0.3],
            "continuous_action_institution": [0.1, -0.2, 0.3, -0.4],
            "continuous_action_retail": [-0.2, 0.1, -0.4, 0.3],
        }
    )
    config = {
        "investors": ["foreign", "institution", "retail"],
        "features": {
            "selected": ["momentum", "herd"],
            "params": {"momentum": {"window": 1}, "herd": {"window": 1}},
        },
    }

    features, names = build_state_feature_tensor(frame, config)
    valid = valid_rows_for_continuous_model(frame, features, config["investors"])

    assert names == ["momentum", "herd"]
    assert features.shape == (4, 3, 2)
    assert not valid[0]
    np.testing.assert_allclose(features[2, 0], [np.log(103.0 / 101.0), -0.15])


def test_continuous_policy_outputs_clipped_state_score_and_l1():
    model = ContinuousInvestorIRLModel(num_features=2, num_contexts=1)
    model.beta.data.copy_(torch.tensor([0.6, 0.6]))
    model.context_weights.data.copy_(torch.tensor([[0.2], [-0.4]]))
    features = torch.tensor([[1.0, 1.0], [-2.0, 0.0]])
    context = torch.tensor([[1.0], [1.0]])

    output = model(features, context)

    torch.testing.assert_close(output["state_score"], torch.tensor([1.0, -1.6]))
    torch.testing.assert_close(output["action"], torch.tensor([1.0, -1.0]))
    assert np.isclose(model.l1_penalty().item(), 1.8)
