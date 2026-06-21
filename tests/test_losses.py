import torch

from src.training.losses import behavior_nll, regularized_loss


def test_single_investor_loss_has_no_investor_sum():
    logits = torch.zeros(4, 2)
    labels = torch.ones(4, dtype=torch.long)
    nll = behavior_nll(logits, labels)
    total = regularized_loss(nll, torch.tensor(2.0), lambda_l1=0.1)

    assert nll.ndim == 0
    assert torch.isclose(total, nll + 0.2)
