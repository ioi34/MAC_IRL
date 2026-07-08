from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.training.checkpoint import save_checkpoint
from src.training.trainer import resolve_device


def _continuous_weight_history_rows(
    model,
    epoch: int,
    feature_names: list[str],
    context_names: list[str],
) -> list[dict]:
    rows = [
        {
            "epoch": epoch,
            "parameter": "beta",
            "feature": feature,
            "context": "",
            "weight": float(weight),
        }
        for feature, weight in zip(feature_names, model.beta.detach().cpu().tolist(), strict=True)
    ]
    if context_names and model.context_weights is not None:
        context_weights = model.context_weights.detach().cpu()
        for feature_idx, feature in enumerate(feature_names):
            for context_idx, context in enumerate(context_names):
                rows.append(
                    {
                        "epoch": epoch,
                        "parameter": "B",
                        "feature": feature,
                        "context": context,
                        "weight": float(context_weights[feature_idx, context_idx]),
                    }
                )
    return rows


def train_continuous_investor_model(
    model,
    train_loader: DataLoader,
    train_config: dict,
    checkpoint_path: str | Path,
    feature_names: list[str],
    context_names: list[str] | None = None,
) -> dict[str, float]:
    device = resolve_device(train_config.get("device", "auto"))
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
    )
    lambda_l1 = float(train_config["loss"]["lambda_l1"])
    final_mse = 0.0

    checkpoint_path = Path(checkpoint_path)
    history_path = checkpoint_path.with_name(checkpoint_path.stem + "_loss_history.csv")
    weight_history_path = checkpoint_path.with_name(
        checkpoint_path.stem + "_weight_history.csv"
    )
    context_names = context_names or []
    history_rows: list[dict] = []
    weight_history_rows = _continuous_weight_history_rows(
        model,
        epoch=0,
        feature_names=feature_names,
        context_names=context_names,
    )

    for epoch in range(1, int(train_config["epochs"]) + 1):
        model.train()
        total_mse = 0.0
        total_samples = 0
        for batch in train_loader:
            if len(batch) == 2:
                features, target = batch
                context = None
            else:
                features, context, target = batch
                context = context.to(device)
            features = features.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features, context)["action"]
            mse = F.mse_loss(prediction, target)
            l1_val = model.l1_penalty()
            loss = mse + lambda_l1 * l1_val
            loss.backward()
            optimizer.step()
            total_mse += float(mse.detach()) * len(target)
            total_samples += len(target)

        final_mse = total_mse / total_samples
        l1_value = float(model.l1_penalty().detach())
        history_rows.append(
            {
                "epoch": epoch,
                "train_mse": round(final_mse, 6),
                "train_rmse": round(final_mse**0.5, 6),
                "l1_penalty": round(l1_value, 6),
                "total_loss": round(final_mse + lambda_l1 * l1_value, 6),
            }
        )
        weight_history_rows.extend(
            _continuous_weight_history_rows(
                model,
                epoch=epoch,
                feature_names=feature_names,
                context_names=context_names,
            )
        )

    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_mse", "train_rmse", "l1_penalty", "total_loss"],
        )
        writer.writeheader()
        writer.writerows(history_rows)
    with weight_history_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "parameter", "feature", "context", "weight"],
        )
        writer.writeheader()
        writer.writerows(weight_history_rows)

    metrics = {"train_mse": final_mse, "train_rmse": final_mse**0.5}
    save_checkpoint(checkpoint_path, model, optimizer, epoch, metrics)
    return metrics
