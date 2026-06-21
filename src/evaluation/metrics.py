from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss


def evaluate_logits(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    predicted = probabilities.argmax(axis=1)
    true = labels.detach().cpu().numpy()
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, average="macro", labels=[0, 1], zero_division=0)),
        "nll": float(log_loss(true, probabilities, labels=[0, 1])),
        "confusion_matrix": confusion_matrix(true, predicted, labels=[0, 1]).tolist(),
    }
