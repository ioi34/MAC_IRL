from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.training.checkpoint import save_checkpoint
from src.training.losses import behavior_nll, regularized_loss


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_investor_model(
    model,
    train_loader: DataLoader,
    train_config: dict,
    checkpoint_path: str | Path,
) -> dict[str, float]:
    device = resolve_device(train_config.get("device", "auto"))
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
    )
    lambda_l1 = float(train_config["loss"]["lambda_l1"])
    final_nll = 0.0

    for epoch in range(1, int(train_config["epochs"]) + 1):
        model.train()
        total_nll = 0.0
        total_samples = 0
        for phi, labels in train_loader:
            phi = phi.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            nll = behavior_nll(model(phi)["logits"], labels)
            loss = regularized_loss(nll, model.l1_penalty(), lambda_l1)
            loss.backward()
            optimizer.step()
            total_nll += float(nll.detach()) * len(labels)
            total_samples += len(labels)
        final_nll = total_nll / total_samples

    metrics = {"train_nll": final_nll}
    save_checkpoint(checkpoint_path, model, optimizer, epoch, metrics)
    return metrics
