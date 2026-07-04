# 기관 lambda_l1 재탐색 (relative 추가 후)

- 날짜/시간: 2026-07-04 16:20
- 목적: relative feature 추가 후 기관의 lambda_l1을 재탐색하여 relative β가 살아나는지 확인
- 변경한 것: institution lambda_l1만 0.01 → 0.005, 0.003, 0.001 탐색 (외국인·개인 HP 고정)
- 고정 조건: 5개 피처(underwater, herd, momentum, relative, volatility) / kospi_return_1d / 973샘플 / CPCV 45 split
- 설정 파일: `configs/experiment_inst_l1_0005.yaml`, `configs/experiment_inst_l1_0003.yaml`, `configs/experiment_inst_l1_0001.yaml`
- 결과 폴더: `runs/inst_l1_0005/`, `runs/inst_l1_0003/`, `runs/inst_l1_0001/`

## 성능 비교 (기관)

| lambda_l1 | accuracy | NLL | macro_F1 |
|-----------|---------|-----|---------|
| 0.010 (기존) | 0.5215 | 0.6927 | 0.5113 |
| 0.005 | 0.5233 | 0.6926 | 0.5135 |
| 0.003 | 0.5223 | 0.6926 | 0.5127 |
| 0.001 | 0.5243 | 0.6926 | 0.5146 |

## 가중치 결과 (기관 relative β)

| lambda_l1 | relative β | 방향 일관성 | 출처 |
|-----------|-----------|-----------|------|
| 0.010 | -0.0037 | 73% | runs/relative_e1/reward_weights_summary.csv |
| 0.005 | -0.0035 | 71% | runs/inst_l1_0005/reward_weights_summary.csv |
| 0.003 | -0.0034 | 73% | runs/inst_l1_0003/reward_weights_summary.csv |
| 0.001 | -0.0034 | 71% | runs/inst_l1_0001/reward_weights_summary.csv |

## 결론

- lambda_l1을 0.001까지 낮춰도 relative β는 -0.003~-0.004에서 변화 없음
- 원인은 정규화 과강이 아니라, 기관 행동이 실제로 relative로 설명되지 않기 때문
- 기관은 삼성-KOSPI200 상대수익률에 반응하지 않음 (벤치마크 추적보다 자체 기준으로 행동)
- 재탐색 실익 없음 → lambda_l1=0.01 원복, relative_e1을 새 베이스라인으로 확정

## 최종 베이스라인 확정

- 설정: `configs/experiment_relative.yaml` (기관 lambda_l1=0.01 유지)
- 결과: `runs/relative_e1/`
- 보상특징 5개: underwater, herd, momentum, relative, volatility

## 근거 구분

- 실제 파일 기반: `runs/inst_l1_*/reward_weights_summary.csv`, `cv_metrics_summary.csv`
