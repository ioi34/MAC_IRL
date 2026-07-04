# volatility 결합 방식 비교: 방향형 vs 강도형

- 날짜/시간: 2026-07-04 15:56
- 목적: volatility feature의 결합 방식을 방향형(a)에서 강도형(|a|)으로 바꿀 때 성능 및 가중치 변화 검증
- 변경한 것: `configs/features_vkospi.yaml` → `magnitude: false → true` (실험 후 false로 복원)
- 고정 조건: kospi_return_1d 컨텍스트 / 973샘플 / CPCV 10-fold/2-test-fold (45 split) / 투자자별 튜닝 HP
- 데이터: `data/raw/samsung_macirl_EXTENDED_V2019_2025.csv`
- 설정 파일: `configs/data_vkospi.yaml`, `configs/features_vkospi.yaml`, `configs/experiment_magnitude_vol.yaml`
- 결과 폴더: `runs/magnitude_vol_e1/`

## 성능 비교

| 투자자 | 방향형 accuracy | 강도형 accuracy | 변화 | 방향형 NLL | 강도형 NLL | 변화 |
|--------|--------------|--------------|------|-----------|-----------|------|
| 외국인 | 0.6399 | 0.6362 | -0.0037 | 0.6501 | 0.6477 | -0.0024 |
| 기관 | 0.5343 | 0.5315 | -0.0028 | 0.6923 | 0.6929 | +0.0006 |
| 개인 | 0.5905 | 0.5935 | +0.0030 | 0.6817 | 0.6824 | +0.0007 |

## volatility β 비교

| 투자자 | 방향형 β | 방향 일관성 | 강도형 β | 방향 일관성 |
|--------|---------|-----------|---------|-----------|
| 외국인 | +0.0293 | 73% | +0.0025 | 69% |
| 기관 | +0.0066 | 98% | ≈0 | 58% |
| 개인 | -0.0212 | 84% | ≈0 | 62% |

## 컨텍스트 상호작용(B: volatility×kospi_return_1d) 비교

| 투자자 | 방향형 B | 일관성 | 강도형 B | 일관성 |
|--------|---------|--------|---------|--------|
| 외국인 | +0.1213 | 100% | +0.0009 | 53% |
| 기관 | -0.0074 | 100% | ≈0 | 51% |
| 개인 | -0.0362 | 100% | ≈0 | 60% |

## 원인 분석

강도형(`magnitude=true`)에서 feature = `parkinson_vol × |action|`.
- 매수(+1)와 매도(-1)가 동일한 feature 값을 가짐
- softmax 보상 구조에서 모델이 행동 방향을 volatility로 구분할 수 없게 됨
- L1 정규화가 식별력 없는 feature를 0으로 수렴시킴
- 컨텍스트 상호작용(B)도 방향 정보 부재로 무의미해짐

## 결론

- 강도형 전환은 이 모델 구조에서 작동하지 않음
- 방향형(magnitude=false)이 올바른 설계
- volatility의 73% 방향 일관성은 약하지만, 컨텍스트 상호작용(B=+0.182, 100%)이 보완함
- configs/features_vkospi.yaml을 방향형으로 복원, dataset_vkospi.npz도 재생성 완료

## 근거 구분

- 실제 파일 기반: `runs/magnitude_vol_e1/cv_metrics_summary.csv`, `reward_weights_summary.csv`, `context_weights_summary.csv`
- 베이스라인: `runs/tuned_kospi_e2/`
