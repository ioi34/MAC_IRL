from __future__ import annotations

import torch


def reward_logits(rewards: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    if tau <= 0:
        raise ValueError("tau must be positive")
    return rewards / tau


def action_probabilities(rewards: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    return torch.softmax(reward_logits(rewards, tau=tau), dim=-1)

