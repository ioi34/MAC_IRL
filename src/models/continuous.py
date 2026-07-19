from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn


def build_context_mask(
    feature_names: Sequence[str],
    context_names: Sequence[str],
    interactions: Mapping[str, Sequence[str]] | None,
) -> torch.Tensor | None:
    if interactions is None:
        return None

    unknown_contexts = sorted(set(interactions).difference(context_names))
    missing_contexts = sorted(set(context_names).difference(interactions))
    if unknown_contexts or missing_contexts:
        raise ValueError(
            "context_interactions must define exactly the selected contexts; "
            f"unknown={unknown_contexts}, missing={missing_contexts}"
        )

    feature_to_index = {name: idx for idx, name in enumerate(feature_names)}
    context_to_index = {name: idx for idx, name in enumerate(context_names)}
    mask = torch.zeros((len(feature_names), len(context_names)), dtype=torch.float32)
    for context, features in interactions.items():
        unknown_features = sorted(set(features).difference(feature_names))
        if unknown_features:
            raise ValueError(
                f"Unknown features {unknown_features} in context_interactions[{context!r}]"
            )
        for feature in features:
            mask[feature_to_index[feature], context_to_index[context]] = 1.0
    return mask


class ContinuousInvestorIRLModel(nn.Module):
    """Closed-form continuous-action policy with kappa fixed to 1."""

    def __init__(
        self,
        num_features: int,
        num_contexts: int = 0,
        context_mask: torch.Tensor | None = None,
        context_main_effect: bool = False,
    ):
        super().__init__()
        self.num_contexts = num_contexts
        self.beta = nn.Parameter(torch.zeros(num_features))
        if num_contexts:
            self.context_weights = nn.Parameter(torch.zeros(num_features, num_contexts))
            if context_mask is None:
                context_mask = torch.ones((num_features, num_contexts), dtype=torch.float32)
            if tuple(context_mask.shape) != (num_features, num_contexts):
                raise ValueError(
                    f"context_mask shape {tuple(context_mask.shape)} does not match "
                    f"({num_features}, {num_contexts})"
                )
            self.register_buffer("context_mask", context_mask.detach().clone().float())
            if context_main_effect:
                self.context_main = nn.Parameter(torch.zeros(num_contexts))
            else:
                self.register_parameter("context_main", None)
        else:
            if context_mask is not None:
                raise ValueError("context_mask requires at least one context")
            if context_main_effect:
                raise ValueError("context_main_effect requires at least one context")
            self.register_parameter("context_weights", None)
            self.register_buffer("context_mask", None)
            self.register_parameter("context_main", None)

    def effective_context_weights(self) -> torch.Tensor | None:
        if self.context_weights is None:
            return None
        return self.context_weights * self.context_mask

    def weights(self, context: torch.Tensor | None = None) -> torch.Tensor:
        if self.num_contexts:
            if context is None:
                raise ValueError("context is required for a contextual continuous model")
            return self.beta + context @ self.effective_context_weights().T
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
        if self.context_main is not None:
            state_score = state_score + context @ self.context_main
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
            penalty = penalty + self.effective_context_weights().abs().sum()
        if self.context_main is not None:
            penalty = penalty + self.context_main.abs().sum()
        return penalty
