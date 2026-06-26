from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.training.checkpoint import save_checkpoint
from src.training.losses import behavior_nll, regularized_loss


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
) -> dict[str, float]:
    # 장치 결정 후 모델을 그 장치로 이동
    device = resolve_device(train_config.get("device", "auto"))
    model.to(device)
    # 경사하강법 옵티마이저(Adam). 학습 대상은 model.parameters()(= 보상 가중치 beta)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_config["learning_rate"]),          # 학습률
        weight_decay=float(train_config.get("weight_decay", 0.0)),  # L2 정규화(선택)
    )
    lambda_l1 = float(train_config["loss"]["lambda_l1"])  # L1 페널티 강도
    final_nll = 0.0  # 마지막 epoch의 평균 손실을 기록용으로 보관

    # epoch 단위 반복: 전체 데이터를 epochs 횟수만큼 학습
    for epoch in range(1, int(train_config["epochs"]) + 1):
        model.train()        # 학습 모드로 전환
        total_nll = 0.0      # 이번 epoch의 손실 누적 합
        total_samples = 0    # 이번 epoch의 샘플 수 누적
        # 미니배치 단위 반복: phi=feature 텐서, labels=실제 행동
        for phi, labels in train_loader:
            phi = phi.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)  # 이전 배치의 gradient 초기화
            nll = behavior_nll(model(phi)["logits"], labels)        # 1) 손실 계산
            loss = regularized_loss(nll, model.l1_penalty(), lambda_l1)  # 2) L1 정규화 추가
            loss.backward()    # 3) 역전파: 손실에 대한 gradient 계산
            optimizer.step()   # 4) gradient로 파라미터(beta) 갱신
            # 평균 NLL을 구하기 위해 (배치 손실 × 배치 크기)를 누적
            total_nll += float(nll.detach()) * len(labels)
            total_samples += len(labels)
        final_nll = total_nll / total_samples  # 샘플 가중 평균 손실

    # 마지막 손실을 metrics로 남기고, 모델/옵티마이저 상태를 체크포인트로 저장
    metrics = {"train_nll": final_nll}
    save_checkpoint(checkpoint_path, model, optimizer, epoch, metrics)
    return metrics
