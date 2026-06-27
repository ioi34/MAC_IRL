from __future__ import annotations

import torch
from torch import nn


class FixedLinearReward(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        return torch.einsum("baf,f->ba", phi, self.beta)

    def l1_penalty(self) -> torch.Tensor:
        return self.beta.abs().sum()


class ContextualLinearReward(nn.Module):
    def __init__(self, num_features: int, num_contexts: int):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.context_weights = nn.Parameter(torch.zeros(num_features, num_contexts))

    def forward(self, phi: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        weights = self.beta + context @ self.context_weights.T
        return torch.einsum("baf,bf->ba", phi, weights)

    def l1_penalty(self) -> torch.Tensor:
        return self.beta.abs().sum() + self.context_weights.abs().sum()
