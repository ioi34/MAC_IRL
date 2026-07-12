# 연속 순매수 행동과 바이너리 행동 비교 평가

- 날짜/시간: 2026-07-10 16:27 KST
- 목적: 연속 순매수 강도 모델과 기존 바이너리 모델을 공통 타깃으로 다시 평가해 현재 우열과 다음 개선 방향을 판단한다.
- 변경한 것: 새 학습은 수행하지 않았다. 기존 45개 CPCV OOS 예측을 날짜·split·투자자별로 결합해 공통 타깃으로 재채점했다. 바이너리 확률은 각 split 학습구간의 클래스별 평균 연속 행동으로만 환산해 연속 타깃에 투영했다.
- 고정 조건: 973표본(2022-01-06~2025-12-29), underwater·herd·momentum·volatility, KOSPI200 1일 수익률·FX 252일 수준 컨텍스트, 동일 CPCV 45 split·seed 42.
- 비교 한계: 바이너리는 과거 252일 중앙값 상·하위 라벨이고 연속은 실제 순매수 비율의 0 부호·강도를 학습한다. 학습 목적과 하이퍼파라미터도 동일하지 않으므로 단순 원 지표 비교는 하지 않았다.
- 데이터: `data/processed/dataset_final.npz`, `data/processed/dataset_continuous_final.npz`
- 설정 파일: `runs/final_reward_context/config_snapshot.yaml`, `runs/continuous_net_buy_behavior/config_snapshot.yaml`
- 결과 폴더: `runs/final_reward_context`, `runs/continuous_net_buy_behavior`
- 주요 결과: 연속 타깃에서 외국인은 바이너리 투영이 근소 우세, 개인은 연속이 우세, 기관은 두 모델 모두 평균 예측 수준이다. 중앙값 분류 타깃은 바이너리가 전체 Macro F1에서 우세하다. 따라서 현재 성능만으로 연속 모델의 전면 우세를 선언할 수 없지만, 실제 순매수 강도를 연구하려면 연속 경로를 주모형으로 개선하는 것이 타당하다.

## 공통 연속 타깃 재평가

| 투자자 | 연속 RMSE | 바이너리 투영 RMSE | 연속 방향정확도 | 바이너리 투영 방향정확도 | 연속 상관 | 바이너리 투영 상관 | 판단 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 외국인 | 0.244013 | 0.243223 | 0.614865 | 0.634110 | 0.282297 | 0.329141 | 바이너리 근소 우세 |
| 기관 | 0.124611 | 0.125436 | 0.534066 | 0.541248 | 0.077149 | 0.091804 | 사실상 동률·둘 다 약함 |
| 개인 | 0.315772 | 0.323078 | 0.603878 | 0.597613 | 0.256808 | 0.243589 | 연속 우세 |

- 학습구간 평균 예측 대비 연속 RMSE 개선율: 외국인 6.04%, 기관 0.51%, 개인 6.31%.
- OOS 예측 표준편차/실제 표준편차: 외국인 0.390, 기관 0.069, 개인 0.265. 세 모델 모두 강도 변동을 크게 축소하며 특히 기관은 거의 평균만 예측한다.
- 출처: `runs/continuous_net_buy_behavior/predictions.csv`, `runs/final_reward_context/plots_binary/reconstructed_predictions.csv`, 각 `split_XX/indices.npz`.

## 공통 중앙값 분류 타깃 재평가

| 투자자 | 바이너리 Accuracy | 연속 임계화 Accuracy | 바이너리 Macro F1 | 연속 임계화 Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| 외국인 | 0.631753 | 0.629627 | 0.615702 | 0.583768 |
| 기관 | 0.533771 | 0.518269 | 0.523756 | 0.387852 |
| 개인 | 0.592464 | 0.606934 | 0.578475 | 0.539200 |
| 3투자자 평균 | 0.585996 | 0.584943 | 0.572644 | 0.503607 |

- Accuracy 평균은 비슷하지만 Macro F1은 바이너리가 0.0690 높다.
- 연속 임계화 예측의 매수 비율은 외국인 65.3%, 기관 93.1%, 개인 30.4%로 치우쳤다. 연속 출력의 과소분산과 투자자별 이동 중앙값이 결합한 결과다.
- 기존 바이너리 라벨과 실제 순매수 0 부호의 일치율은 외국인 87.9%, 기관 93.5%, 개인 92.4%다. 따라서 두 라벨은 유사하지만 동일한 buy/sell 정의는 아니다.

## 가중치 결과

두 목적함수의 계수 크기 척도는 다르므로 부호와 split 일관성만 비교한다.

| 대상 | 피처/가중치 | 연속 값 | 연속 일관성 | 바이너리 값 | 바이너리 일관성 | 출처 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| foreign | beta underwater | -0.003482 | 0.644444 | -0.057869 | 0.911111 | 각 모델 `reward_weights_summary.csv` |
| foreign | beta herd | -0.064933 | 1.000000 | -0.231670 | 1.000000 | 각 모델 `reward_weights_summary.csv` |
| foreign | beta momentum | 0.036759 | 1.000000 | 0.139340 | 1.000000 | 각 모델 `reward_weights_summary.csv` |
| foreign | beta volatility | -0.019096 | 0.933333 | 0.006814 | 0.533333 | 각 모델 `reward_weights_summary.csv` |
| institution | beta underwater | -0.001049 | 0.911111 | -0.006162 | 0.622222 | 각 모델 `reward_weights_summary.csv` |
| institution | beta herd | -0.004210 | 1.000000 | -0.047678 | 1.000000 | 각 모델 `reward_weights_summary.csv` |
| institution | beta momentum | -0.000724 | 0.733333 | -0.060825 | 0.977778 | 각 모델 `reward_weights_summary.csv` |
| institution | beta volatility | 0.002645 | 0.911111 | 0.041390 | 0.933333 | 각 모델 `reward_weights_summary.csv` |
| retail | beta underwater | 0.030660 | 1.000000 | 0.103063 | 1.000000 | 각 모델 `reward_weights_summary.csv` |
| retail | beta herd | -0.033370 | 1.000000 | -0.168692 | 1.000000 | 각 모델 `reward_weights_summary.csv` |
| retail | beta momentum | -0.030995 | 1.000000 | -0.058475 | 0.977778 | 각 모델 `reward_weights_summary.csv` |
| retail | beta volatility | 0.008269 | 0.688889 | -0.048813 | 0.888889 | 각 모델 `reward_weights_summary.csv` |

- 기본 beta는 12개 중 10개가 같은 부호이고, 양쪽 모두 일관성 80% 이상인 동일 부호는 7개다.
- 컨텍스트 B는 24개 중 19개가 같은 부호이며, 양쪽 모두 일관성 80% 이상인 동일 부호는 12개다. 변동성 관련 일부 계수는 뒤집혀 잠정 결과로 취급한다.
- beta 출처: `runs/continuous_net_buy_behavior/reward_weights_summary.csv`, `runs/final_reward_context/reward_weights_summary.csv`.
- B 출처: `runs/continuous_net_buy_behavior/context_weights_summary.csv`, `runs/final_reward_context/context_weights_summary.csv`.

- 해석: herd와 momentum을 중심으로 한 핵심 행동 방향은 라벨 정의를 바꿔도 상당히 유지된다. 반면 연속 모델의 강도 예측력은 아직 낮고 2025년 방향정확도도 외국인 0.535, 기관 0.539, 개인 0.548로 약화됐다.
- 다음 액션: (1) 실제 0 부호 바이너리 대조군을 추가하고, (2) 방향 분류+절대 강도 회귀의 hurdle/multitask 모델을 시험하며, (3) 연속 전용 nested tuning과 Huber·가중 손실을 비교하고, (4) 연도별 walk-forward와 3개 이상 seed로 검증한다.

## 근거 구분

- 실제 파일 기반: 위 `runs/` OOS 예측·가중치·split 파일과 `data/processed/` 배열.
- Notion/GitHub 메모만으로 판단한 항목: 없음.
- 코드·설정에서 추론한 항목: 바이너리 라벨 의미와 연속 라벨 의미의 차이. 공통 타깃 수치는 실제 예측 파일을 재채점해 계산했다.
