from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.training.checkpoint import save_checkpoint
from src.training.losses import behavior_nll, regularized_loss


def _weight_history_rows(
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
        for feature, weight in zip(
            feature_names,
            model.reward_model.beta.detach().cpu().tolist(),
            strict=True,
        )
    ]
    if context_names:
        context_weights = model.reward_model.context_weights.detach().cpu()
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


def resolve_device(name: str) -> torch.device:
    # 학습에 쓸 장치를 결정한다.
    # "auto"가 아니면 지정된 장치를 그대로 사용.
    if name != "auto":
        return torch.device(name)
    # "auto"면 가능한 가속기를 우선순위대로 선택: CUDA(GPU) > MPS(애플 실리콘) > CPU
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
    log_every = int(train_config.get("log_every", 0))
    final_nll = 0.0

    checkpoint_path = Path(checkpoint_path)
    history_path = checkpoint_path.with_name(checkpoint_path.stem + "_loss_history.csv")
    weight_history_path = checkpoint_path.with_name(
        checkpoint_path.stem + "_weight_history.csv"
    )
    history_rows: list[dict] = []
    context_names = context_names or []
    weight_history_rows = _weight_history_rows(
        model,
        epoch=0,
        feature_names=feature_names,
        context_names=context_names,
    )

    for epoch in range(1, int(train_config["epochs"]) + 1):
        model.train()
        total_nll = 0.0
        total_samples = 0
        for batch in train_loader:
            if len(batch) == 2:
                phi, labels = batch
                context = None
            else:
                phi, context, labels = batch
                context = context.to(device)
            phi = phi.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            nll = behavior_nll(model(phi, context)["logits"], labels)
            loss = regularized_loss(nll, model.l1_penalty(), lambda_l1)
            loss.backward()
            optimizer.step()
            total_nll += float(nll.detach()) * len(labels)
            total_samples += len(labels)
        final_nll = total_nll / total_samples
        l1_val = float(model.l1_penalty().detach())
        history_rows.append({
            "epoch": epoch,
            "train_nll": round(final_nll, 6),
            "l1_penalty": round(l1_val, 6),
            "total_loss": round(final_nll + lambda_l1 * l1_val, 6),
        })
        weight_history_rows.extend(
            _weight_history_rows(
                model,
                epoch=epoch,
                feature_names=feature_names,
                context_names=context_names,
            )
        )

    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_nll", "l1_penalty", "total_loss"])
        writer.writeheader()
        writer.writerows(history_rows)
    with weight_history_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "parameter", "feature", "context", "weight"],
        )
        writer.writeheader()
        writer.writerows(weight_history_rows)

    metrics = {"train_nll": final_nll}
    save_checkpoint(checkpoint_path, model, optimizer, epoch, metrics)
    return metrics
