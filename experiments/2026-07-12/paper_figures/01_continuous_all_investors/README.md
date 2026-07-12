# 연속 순매수: 전체 투자자

외국인·기관·개인의 연속 행동

\[
a_t = \frac{Buy_t-Sell_t}{Buy_t+Sell_t}
\]

에 대한 CPCV OOS 실제값과 예측값이다.

## 그림

| 파일 | 설명 |
| --- | --- |
| `fig_01_all_investors_timeseries.png` | 세 투자자 통합 시계열 |
| `fig_02_all_investors_scatter.png` | 실제값과 예측값 산점도 |
| `fig_03_foreign_timeseries.png` | 외국인 시계열 |
| `fig_04_institution_timeseries.png` | 기관 시계열 |
| `fig_05_retail_timeseries.png` | 개인 시계열 |

## 주요 수치

| 투자자 | RMSE | 방향 정확도 | Pearson 상관 |
| --- | ---: | ---: | ---: |
| 외국인 | 0.244013 | 0.614865 | 0.282297 |
| 기관 | 0.124611 | 0.534066 | 0.077149 |
| 개인 | 0.315772 | 0.603878 | 0.256808 |

- 원본: `runs/continuous_net_buy_behavior/plots/`
- 원자료: `data/averaged_predictions_by_date.csv`
- 평가표: `data/cv_metrics_summary.csv`
- 해석: 기관의 낮은 RMSE는 실제 행동 진폭이 작은 영향이 크며, 방향성과 상관은 세 투자자 중 가장 약하다.
