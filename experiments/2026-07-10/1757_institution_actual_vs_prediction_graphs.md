# 기관 연속 행동 실제값-예측값 비교 그래프

- 날짜/시간: 2026-07-10 17:57
- 목적: 기관 연속 순매수 행동 모델의 예측 결과를 실제 행동과 직접 비교하는 그래프 생성.
- 변경한 것: 새 학습 없음. 기존 CPCV `predictions.csv`를 날짜별로 평균 집계한 뒤 실제값과 비교.
- 고정 조건: 기관 투자자만 사용, 973개 거래일, 각 날짜는 CPCV 9개 test path 예측 평균으로 집계.
- 데이터: `samsung_macirl_EXTENDED_V2019_2025.csv` 기반 처리 데이터 및 기존 ablation run.
- 설정 파일: 기존 `runs/continuous_institution_ablation/*/config_snapshot.yaml`
- 결과 폴더: `experiments/2026-07-10/institution_prediction_graphs`
- 주요 결과:
  - `E1 persistence`와 `E7 persistence x liquidity`는 baseline보다 실제값과의 상관 및 방향 정확도가 높다.
  - 다만 예측 표준편차/실제 표준편차가 약 0.09 수준이라, 실제 기관 행동의 크기 변동을 충분히 복원하지 못한다.
  - `E7`은 방향 정확도는 가장 높지만, `E1` 대비 개선폭이 작아 본문 주력 그림은 `E1`, 부록 비교는 `E7`이 적절하다.

## 생성 파일

| 파일 | 내용 |
| --- | --- |
| `experiments/2026-07-10/institution_prediction_graphs/institution_actual_vs_prediction_raw.png` | 실제값과 예측값 raw 시계열 |
| `experiments/2026-07-10/institution_prediction_graphs/institution_actual_vs_prediction_rolling_20d.png` | 20일 이동평균 시계열 |
| `experiments/2026-07-10/institution_prediction_graphs/institution_actual_vs_prediction_scatter.png` | 실제값-예측값 산점도 |
| `experiments/2026-07-10/institution_prediction_graphs/institution_actual_vs_prediction_aggregated.csv` | 날짜별 집계 예측값 |
| `experiments/2026-07-10/institution_prediction_graphs/institution_actual_vs_prediction_metrics.csv` | 날짜별 집계 기준 비교 지표 |

## 실제값-예측값 비교 지표

| 모델 | RMSE | Pearson | 방향 정확도 | Balanced Acc. | 예측/실제 표준편차 | 출처 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| E0 baseline | 0.125153 | 0.098309 | 0.543679 | 0.537489 | 0.074505 | `institution_actual_vs_prediction_metrics.csv` |
| E1 persistence | 0.124491 | 0.152569 | 0.549846 | 0.545299 | 0.091725 | `institution_actual_vs_prediction_metrics.csv` |
| E7 persistence x liquidity | 0.124482 | 0.152534 | 0.558068 | 0.552804 | 0.092922 | `institution_actual_vs_prediction_metrics.csv` |

## 가중치 결과

| 대상 | 피처/가중치 | 값 | 변동성/일관성 | 출처 |
| --- | --- | ---: | ---: | --- |
| E0 baseline / institution | herd | -0.004210 | std 0.000589 / 부호일관성 1.000 | `experiments/2026-07-10/institution_current_data_analysis/institution_reward_weights.csv` |
| E0 baseline / institution | volatility | 0.002645 | std 0.001865 / 부호일관성 0.911 | `experiments/2026-07-10/institution_current_data_analysis/institution_reward_weights.csv` |
| E1 persistence / institution | execution_persistence_3 | 0.004773 | std 0.000062 / 부호일관성 1.000 | `experiments/2026-07-10/institution_current_data_analysis/institution_reward_weights.csv` |
| E7 persistence x liquidity / institution | execution_persistence_3 | 0.004770 | std 0.000062 / 부호일관성 1.000 | `experiments/2026-07-10/institution_current_data_analysis/institution_reward_weights.csv` |
| E7 persistence x liquidity / institution | execution_persistence_3 x liquidity_capacity_20 | 0.001011 | std 0.001091 / 부호일관성 0.911 | `experiments/2026-07-10/institution_current_data_analysis/institution_context_weights.csv` |

- 해석: 실제 행동은 급격하고 큰 음/양 순매수 변동이 많은 반면, 모델은 부호와 완만한 흐름은 일부 따라가지만 예측값의 진폭이 작다. 따라서 현재 개선은 “방향성 설명력 개선”이지 “행동 크기 복원 완료”로 보기는 어렵다.
- 다음 액션: 본문에는 `E1 persistence`의 20일 이동평균 그래프와 산점도를 제시하고, under-dispersion 문제를 한계 및 향후 개선 방향으로 명시한다.
