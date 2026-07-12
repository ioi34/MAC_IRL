# 기관 학습조건 진단

기관 E1의 기존 학습조건과 학습량 확대·L1 제거 조건을 비교한다.

| 모델 | epoch | learning rate | L1 |
| --- | ---: | ---: | ---: |
| E1 기준 | 10 | 0.0005 | 0.01 |
| T2 | 50 | 0.001 | 0.01 |
| T6 | 50 | 0.001 | 0 |

## 그림

| 파일 | 설명 |
| --- | --- |
| `fig_01_actual_vs_hyperparameters_raw.png` | 일별 실제값과 파라미터별 예측 |
| `fig_02_actual_vs_hyperparameters_rolling_20d.png` | 20일 이동평균 비교 |
| `fig_03_actual_vs_hyperparameters_scatter.png` | 실제값과 예측값 산점도 |

## 날짜별 평균 예측 수치

| 모델 | RMSE | Pearson | Balanced Accuracy | 예측/실제 표준편차 |
| --- | ---: | ---: | ---: | ---: |
| E1 기준 | 0.124491 | 0.152569 | 0.545299 | 0.091725 |
| T2 | 0.123892 | 0.171173 | 0.565014 | 0.158751 |
| T6 | 0.125397 | 0.136645 | 0.559728 | 0.257768 |

- 원본: `experiments/2026-07-10/institution_hparam_graphs/`
- 원자료: `data/aggregated_predictions.csv`
- 평가표: `data/metrics.csv`
- 해석: L1 제거는 진폭을 키웠지만 RMSE와 상관을 악화했다. T2는 CPCV에서 개선됐지만 2023 expanding walk-forward에서 과적합되어 주모형으로 채택하지 않았다.
