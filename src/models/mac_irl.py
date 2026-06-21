from __future__ import annotations

import torch
from torch import nn

from src.models.policy import reward_logits
from src.models.reward import FixedLinearReward


class InvestorIRLModel(nn.Module):
    def __init__(self, num_features: int, tau: float = 1.0):
        super().__init__()
        self.tau = tau
        self.reward_model = FixedLinearReward(num_features)

    def forward(self, phi: torch.Tensor) -> dict[str, torch.Tensor]:
        rewards = self.reward_model(phi)
        return {"rewards": rewards, "logits": reward_logits(rewards, tau=self.tau)}

    def l1_penalty(self) -> torch.Tensor:
        return self.reward_model.l1_penalty()
