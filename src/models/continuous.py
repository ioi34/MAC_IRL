from __future__ import annotations

import torch
from torch import nn


class ContinuousInvestorIRLModel(nn.Module):
    """Closed-form continuous-action policy with kappa fixed to 1."""

    def __init__(self, num_features: int, num_contexts: int = 0):
        super().__init__()
        self.num_contexts = num_contexts
        self.beta = nn.Parameter(torch.zeros(num_features))
        if num_contexts:
            self.context_weights = nn.Parameter(torch.zeros(num_features, num_contexts))
        else:
            self.register_parameter("context_weights", None)

    def weights(self, context: torch.Tensor | None = None) -> torch.Tensor:
        if self.num_contexts:
            if context is None:
                raise ValueError("context is required for a contextual continuous model")
            return self.beta + context @ self.context_weights.T
        return self.beta

    def forward(
        self,
        features: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        weights = self.weights(context)
        if weights.ndim == 1:
            state_score = features @ weights
        else:
            state_score = torch.einsum("bf,bf->b", features, weights)
        action = torch.clamp(state_score, -1.0, 1.0)
        reward_at_policy = state_score * action - 0.5 * action.square()
        return {
            "state_score": state_score,
            "action": action,
            "reward_at_policy": reward_at_policy,
        }

    def l1_penalty(self) -> torch.Tensor:
        penalty = self.beta.abs().sum()
        if self.context_weights is not None:
            penalty = penalty + self.context_weights.abs().sum()
        return penalty
