# 연속 순매수 행동 실험

- 날짜/시간: 2026-07-08 23:41 KST
- 목적: PDF의 정의대로 이진/4분류 행동을 대체하지 않고, 별도 연속 행동 경로에서 다음 날 순매수 강도 `[-1, 1]`를 예측한다.
- 변경한 것: `a=(buy_value-sell_value)/(buy_value+sell_value)` 연속 라벨, action-free 상태특징, `q=(β+BC)^T x`, `clip(q,-1,1)` 정책, MSE+L1 학습, 연속행동 평가 지표를 추가했다.
- 고정 조건: 기존 바이너리/4분류 라벨 코드와 기존 `scripts/train.py` 분류 파이프라인은 유지. CPCV, purged/embargo split, train-only feature/context scaling 유지.
- 데이터: `samsung_macirl_EXTENDED_2019_2025.csv`, 973개 샘플, 2022-01-06부터 2025-12-29까지.
- 설정 파일: `configs/data_continuous.yaml`, `configs/features_continuous.yaml`, `configs/experiment_continuous.yaml`, `configs/model.yaml`, `configs/train.yaml`
- 결과 폴더: `runs/continuous_net_buy_behavior`
- 주요 결과: 외국인 MAE 0.1961/RMSE 0.2440/방향정확도 0.6149, 기관 MAE 0.0940/RMSE 0.1246/방향정확도 0.5341, 개인 MAE 0.2652/RMSE 0.3158/방향정확도 0.6039. 세 투자자 모두 포화율은 0.0으로 `clip` 경계에 붙지 않았다.
- 결론 근거: 실제 실행 파일 `runs/continuous_net_buy_behavior/cv_metrics_summary.csv`, `runs/continuous_net_buy_behavior/reward_weights_summary.csv`, `runs/continuous_net_buy_behavior/context_weights_summary.csv` 기반.

## 가중치 결과

| 대상 | 피처/가중치 | 값 | 변동성/일관성 | 출처 |
| --- | --- | ---: | ---: | --- |
| foreign | beta underwater | -0.003482 | std 0.006763 / sign 0.644444 | `runs/continuous_net_buy_behavior/reward_weights_summary.csv` |
| foreign | beta herd | -0.064933 | std 0.010640 / sign 1.000000 | `runs/continuous_net_buy_behavior/reward_weights_summary.csv` |
| foreign | beta momentum | 0.036759 | std 0.005472 / sign 1.000000 | `runs/continuous_net_buy_behavior/reward_weights_summary.csv` |
| foreign | beta volatility | -0.019096 | std 0.013699 / sign 0.933333 | `runs/continuous_net_buy_behavior/reward_weights_summary.csv` |
| institution | beta underwater | -0.001049 | std 0.001016 / sign 0.911111 | `runs/continuous_net_buy_behavior/reward_weights_summary.csv` |
| institution | beta herd | -0.004210 | std 0.000589 / sign 1.000000 | `runs/continuous_net_buy_behavior/reward_weights_summary.csv` |
| institution | beta momentum | -0.000724 | std 0.001265 / sign 0.733333 | `runs/continuous_net_buy_behavior/reward_weights_summary.csv` |
| institution | beta volatility | 0.002645 | std 0.001865 / sign 0.911111 | `runs/continuous_net_buy_behavior/reward_weights_summary.csv` |
| retail | beta underwater | 0.030660 | std 0.001541 / sign 1.000000 | `runs/continuous_net_buy_behavior/reward_weights_summary.csv` |
| retail | beta herd | -0.033370 | std 0.000938 / sign 1.000000 | `runs/continuous_net_buy_behavior/reward_weights_summary.csv` |
| retail | beta momentum | -0.030995 | std 0.000960 / sign 1.000000 | `runs/continuous_net_buy_behavior/reward_weights_summary.csv` |
| retail | beta volatility | 0.008269 | std 0.015846 / sign 0.688889 | `runs/continuous_net_buy_behavior/reward_weights_summary.csv` |

- 컨텍스트 계수 `B`: 전체 결과는 `runs/continuous_net_buy_behavior/context_weights_summary.csv`에 저장했다. 강한 일관성 예시는 외국인 `volatility × fx_level_z_252` 양수 1.000000, 기관 `volatility × fx_level_z_252` 양수 1.000000, 개인 `underwater × fx_level_z_252` 음수 1.000000, 개인 `herd × kospi_return_1d` 음수 1.000000이다.
- 해석: 연속 행동에서는 외국인 momentum이 순매수 방향을 지지하고 herd는 반대 방향으로 안정적이다. 개인은 underwater가 순매수 방향, herd/momentum은 순매도 방향으로 강하게 나타났다. 기관은 행동 진폭 자체가 작아 MAE는 가장 낮지만 방향정확도와 상관은 약하다.
- 다음 액션: 기존 분류 방식과 같은 기간/feature로 성능을 나란히 비교하거나, PDF 정의에 맞춘 상태특징 후보를 더 엄격히 분리해 feature ablation을 진행한다.
