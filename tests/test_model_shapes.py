import torch

from src.models.mac_irl import InvestorIRLModel


def test_independent_investor_model_output_shape():
    model = InvestorIRLModel(num_features=4, tau=1.0)
    phi = torch.randn(7, 4, 4)
    output = model(phi)

    assert output["rewards"].shape == (7, 4)
    assert output["logits"].shape == (7, 4)


def test_investor_models_do_not_share_reward_weights():
    foreign = InvestorIRLModel(num_features=4)
    retail = InvestorIRLModel(num_features=4)

    assert foreign.reward_model.beta.data_ptr() != retail.reward_model.beta.data_ptr()


def test_contextual_model_output_and_weight_shapes():
    model = InvestorIRLModel(num_features=3, num_contexts=4)
    phi = torch.randn(7, 2, 3)
    context = torch.randn(7, 4)

    output = model(phi, context)

    assert output["rewards"].shape == (7, 2)
    assert output["logits"].shape == (7, 2)
    assert model.reward_model.context_weights.shape == (3, 4)


def test_zero_context_weights_match_fixed_linear_model():
    fixed = InvestorIRLModel(num_features=3)
    contextual = InvestorIRLModel(num_features=3, num_contexts=2)
    beta = torch.tensor([0.2, -0.3, 0.4])
    fixed.reward_model.beta.data.copy_(beta)
    contextual.reward_model.beta.data.copy_(beta)
    phi = torch.randn(5, 2, 3)
    context = torch.randn(5, 2)

    fixed_output = fixed(phi)
    contextual_output = contextual(phi, context)

    torch.testing.assert_close(contextual_output["rewards"], fixed_output["rewards"])
    torch.testing.assert_close(contextual_output["logits"], fixed_output["logits"])


def test_contextual_l1_penalty_includes_beta_and_context_weights():
    model = InvestorIRLModel(num_features=2, num_contexts=2)
    model.reward_model.beta.data.copy_(torch.tensor([1.0, -2.0]))
    model.reward_model.context_weights.data.copy_(
        torch.tensor([[3.0, -4.0], [-5.0, 6.0]])
    )

    assert model.l1_penalty().item() == 21.0
