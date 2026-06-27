from __future__ import annotations

import torch
from torch import nn

from src.models.policy import reward_logits
from src.models.reward import ContextualLinearReward, FixedLinearReward


class InvestorIRLModel(nn.Module):
    def __init__(self, num_features: int, tau: float = 1.0, num_contexts: int = 0):
        super().__init__()
        self.tau = tau
        self.num_contexts = num_contexts
        if num_contexts:
            self.reward_model = ContextualLinearReward(num_features, num_contexts)
        else:
            self.reward_model = FixedLinearReward(num_features)

    def forward(
        self,
        phi: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.num_contexts:
            if context is None:
                raise ValueError("context is required for a contextual reward model")
            rewards = self.reward_model(phi, context)
        else:
            rewards = self.reward_model(phi)
        return {"rewards": rewards, "logits": reward_logits(rewards, tau=self.tau)}

    def l1_penalty(self) -> torch.Tensor:
        return self.reward_model.l1_penalty()
