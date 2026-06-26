from __future__ import annotations

import torch
import torch.nn.functional as F


def behavior_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # 행동 복제(behavior cloning) 손실.
    # 모델이 예측한 행동분포(logits)와 실제 투자자 행동(labels)의 교차엔트로피.
    # 값이 작을수록 모델 행동이 실제 행동과 비슷하다는 뜻.
    return F.cross_entropy(logits, labels)


def regularized_loss(
    nll: torch.Tensor,       # 위에서 구한 행동 예측 손실
    l1_penalty: torch.Tensor,  # 보상 가중치 beta의 L1 크기 (sum(|beta|))
    lambda_l1: float,        # L1 페널티 강도 (config에서 지정)
) -> torch.Tensor:
    # 최종 손실 = 예측 손실 + L1 정규화.
    # L1 항이 불필요한 feature의 가중치를 0쪽으로 눌러 희소(sparse)하게 만든다.
    return nll + lambda_l1 * l1_penalty
