# 바이너리 정책 비교 기준

기존 buy/sell 바이너리 모델의 실제 행동과 예측 기대 행동을 정리했다. 연속 순매수 모델과 학습 타깃이 다르므로 원 RMSE를 직접 비교하기보다 방향 정확도와 Macro F1의 기준으로 사용한다.

## 그림

| 파일 | 설명 |
| --- | --- |
| `fig_01_all_investors_expected_action_timeseries.png` | 실제 -1/+1과 예측 기대 행동 |
| `fig_02_all_investors_buy_probability_timeseries.png` | 실제 행동과 매수확률 |
| `fig_03_all_investors_expected_action_scatter.png` | 실제 행동과 기대 행동 산점도 |
| `fig_04_foreign_timeseries.png` | 외국인 바이너리 시계열 |
| `fig_05_institution_timeseries.png` | 기관 바이너리 시계열 |
| `fig_06_retail_timeseries.png` | 개인 바이너리 시계열 |

## 주요 수치

| 투자자 | Accuracy | Macro F1 |
| --- | ---: | ---: |
| 외국인 | 0.631753 | 0.615702 |
| 기관 | 0.533771 | 0.523756 |
| 개인 | 0.592464 | 0.578475 |

- 원본: `runs/final_reward_context/plots_binary/`
- 원자료: `data/averaged_predictions_by_date.csv`
- 평가표: `data/cv_metrics_summary.csv`
