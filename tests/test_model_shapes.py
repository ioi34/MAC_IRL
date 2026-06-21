import torch

from src.models.mac_irl import InvestorIRLModel


def test_independent_investor_model_output_shape():
    model = InvestorIRLModel(num_features=4, tau=1.0)
    phi = torch.randn(7, 2, 4)
    output = model(phi)

    assert output["rewards"].shape == (7, 2)
    assert output["logits"].shape == (7, 2)


def test_investor_models_do_not_share_reward_weights():
    foreign = InvestorIRLModel(num_features=4)
    retail = InvestorIRLModel(num_features=4)

    assert foreign.reward_model.beta.data_ptr() != retail.reward_model.beta.data_ptr()
