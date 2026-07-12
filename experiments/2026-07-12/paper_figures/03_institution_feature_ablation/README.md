# 기관 보상특징 비교: E0·E1·E7

기관 연속형 모델의 기존 보상특징과 거래 지속성·유동성 상호작용을 비교한다.

| 모델 | 구성 |
| --- | --- |
| E0 | 기존 보상특징 4개 |
| E1 | E0 + `execution_persistence_3` |
| E7 | E1 + `execution_persistence_3 × liquidity_capacity_20` |

## 그림

| 파일 | 설명 |
| --- | --- |
| `fig_01_actual_vs_e0_e1_e7_raw.png` | 일별 실제값과 세 모델 예측 |
| `fig_02_actual_vs_e0_e1_e7_rolling_20d.png` | 20일 이동평균 비교 |
| `fig_03_actual_vs_e0_e1_e7_scatter.png` | 실제값과 예측값 산점도 |

## 날짜별 평균 예측 수치

| 모델 | RMSE | Pearson | Balanced Accuracy | 예측/실제 표준편차 |
| --- | ---: | ---: | ---: | ---: |
| E0 | 0.125153 | 0.098309 | 0.537489 | 0.074505 |
| E1 | 0.124491 | 0.152569 | 0.545299 | 0.091725 |
| E7 | 0.124482 | 0.152534 | 0.552804 | 0.092922 |

- 원본: `experiments/2026-07-10/institution_prediction_graphs/`
- 원자료: `data/aggregated_predictions.csv`
- 평가표: `data/metrics.csv`
- 해석: E1은 방향성과 상관을 개선하지만 강도 과소분산은 해결하지 못했다. E7의 E1 대비 개선은 매우 작다.
