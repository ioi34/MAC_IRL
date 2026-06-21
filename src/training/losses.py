from __future__ import annotations

import torch
import torch.nn.functional as F


def behavior_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def regularized_loss(
    nll: torch.Tensor,
    l1_penalty: torch.Tensor,
    lambda_l1: float,
) -> torch.Tensor:
    return nll + lambda_l1 * l1_penalty
