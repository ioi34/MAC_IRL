# 보고서용 가중치 변화·학습 관찰성 실험

- 날짜/시간: 2026-06-30 10:50 KST
- 목적: 기존 MAC-IRL 보고서의 E2 결과를 재현하고, 에폭별 손실과 보상 가중치 변화 데이터를 생성
- 변경한 것: 각 투자자·CPCV split에서 에폭별 `train_nll`, L1, total loss와 `beta`, 컨텍스트 계수 `B`를 CSV로 기록
- 고정 조건: binary 행동, 973표본, underwater·herd(5)·momentum(20)·Parkinson volatility(20)·persist, KOSPI200 1일 수익률 컨텍스트, CPCV 10-fold/2-test-fold(45 split), purge 1, embargo 5, seed 42, Adam lr 0.01, 200 epochs, L1 0.001
- 데이터: `samsung_macirl_EXTENDED_V2019_2025.csv` → `data/processed/dataset_vkospi.npz` (원천 1,719일, 최신 유효 973개)
- 설정 파일: `configs/data_vkospi.yaml`, `configs/features_vkospi.yaml`, `configs/model.yaml`, `configs/train.yaml`, `experiments/2026-06-30/configs/report_observability_vkospi_e2.yaml`
- 결과 폴더: `runs/report_observability_vkospi_e2/`
- 주요 결과: 기존 E2 성능을 재현했으며, 135개 독립 모델 모두 loss history 200행과 weight history 2,010행(초기 0에폭 포함)을 생성

## 행동 재현 성능

| 투자자 | Accuracy | Macro F1 | NLL |
| --- | ---: | ---: | ---: |
| 외국인 | 0.6315 | 0.6145 | 0.6526 |
| 기관 | 0.5281 | 0.5167 | 0.7051 |
| 개인 | 0.5887 | 0.5736 | 0.6894 |

- 출처: `runs/report_observability_vkospi_e2/cv_metrics_summary.csv`

## 학습 관찰성

| 투자자 | 1에폭 NLL | 최종 NLL | 감소량 | 수렴 에폭 평균 |
| --- | ---: | ---: | ---: | ---: |
| 외국인 | 0.6833 | 0.6229 | 0.0603 | 15.8 |
| 기관 | 0.6931 | 0.6846 | 0.0085 | 4.9 |
| 개인 | 0.6876 | 0.6592 | 0.0285 | 9.9 |

- 수렴 기준: 최종 NLL과의 절대 차이가 최초로 0.001 이하가 된 에폭
- 출처: `runs/report_observability_vkospi_e2/loss_summary.csv`, `runs/report_observability_vkospi_e2/loss_trajectory_summary.csv`

## 가중치 변화 예시

아래 값은 45개 CPCV split에서 계산한 `beta` 평균이다.

| 투자자 | 피처 | 0 | 1 | 5 | 10 | 20 | 50 | 100 | 200 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 외국인 | herd | 0.000 | -0.032 | -0.131 | -0.174 | -0.188 | -0.216 | -0.214 | -0.207 |
| 외국인 | momentum | 0.000 | 0.031 | 0.127 | 0.161 | 0.148 | 0.140 | 0.140 | 0.144 |
| 외국인 | underwater | 0.000 | -0.027 | -0.074 | -0.070 | -0.077 | -0.078 | -0.088 | -0.084 |
| 기관 | momentum | 0.000 | -0.024 | -0.067 | -0.069 | -0.064 | -0.065 | -0.070 | -0.064 |
| 기관 | volatility | 0.000 | 0.016 | 0.049 | 0.053 | 0.050 | 0.045 | 0.046 | 0.045 |
| 개인 | herd | 0.000 | -0.030 | -0.108 | -0.122 | -0.137 | -0.138 | -0.139 | -0.137 |
| 개인 | momentum | 0.000 | -0.030 | -0.093 | -0.086 | -0.092 | -0.097 | -0.097 | -0.099 |
| 개인 | persist | 0.000 | 0.030 | 0.098 | 0.096 | 0.097 | 0.087 | 0.090 | 0.087 |
| 개인 | underwater | 0.000 | 0.021 | 0.024 | -0.026 | -0.045 | -0.044 | -0.051 | -0.047 |

- 전체 15개 `beta` 표: `runs/report_observability_vkospi_e2/beta_trajectory_table.csv`
- 전체 15개 `B` 표: `runs/report_observability_vkospi_e2/context_weight_trajectory_table.csv`
- long-form 원본 요약: `runs/report_observability_vkospi_e2/weight_trajectory_summary.csv`

## 가중치 결과

| 투자자 | 피처 | 최종 beta 평균 ± 표준편차 | beta 부호 일관성 | 최종 B 평균 ± 표준편차 | B 부호 일관성 | 출처 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 외국인 | underwater | -0.0836 ± 0.0373 | 100.0% | -0.0657 ± 0.0221 | 100.0% | `reward_weights_summary.csv`, `context_weights_summary.csv` |
| 외국인 | herd | -0.2073 ± 0.0416 | 100.0% | -0.1403 ± 0.0344 | 100.0% | 위와 같음 |
| 외국인 | momentum | 0.1438 ± 0.0335 | 100.0% | -0.1042 ± 0.0284 | 100.0% | 위와 같음 |
| 외국인 | volatility | 0.0337 ± 0.0548 | 73.3% | 0.1824 ± 0.0199 | 100.0% | 위와 같음 |
| 외국인 | persist | 0.0289 ± 0.0256 | 95.6% | -0.1101 ± 0.0350 | 100.0% | 위와 같음 |
| 기관 | underwater | 0.0123 ± 0.0201 | 75.6% | 0.0635 ± 0.0295 | 100.0% | 위와 같음 |
| 기관 | herd | -0.0289 ± 0.0254 | 84.4% | -0.0505 ± 0.0270 | 97.8% | 위와 같음 |
| 기관 | momentum | -0.0643 ± 0.0299 | 100.0% | 0.0193 ± 0.0239 | 82.2% | 위와 같음 |
| 기관 | volatility | 0.0450 ± 0.0255 | 97.8% | -0.0726 ± 0.0200 | 100.0% | 위와 같음 |
| 기관 | persist | 0.0394 ± 0.0376 | 88.9% | -0.0222 ± 0.0248 | 84.4% | 위와 같음 |
| 개인 | underwater | -0.0469 ± 0.0458 | 86.7% | 0.0633 ± 0.0305 | 97.8% | 위와 같음 |
| 개인 | herd | -0.1374 ± 0.0411 | 100.0% | -0.0658 ± 0.0295 | 97.8% | 위와 같음 |
| 개인 | momentum | -0.0985 ± 0.0322 | 100.0% | 0.0523 ± 0.0397 | 86.7% | 위와 같음 |
| 개인 | volatility | -0.0153 ± 0.0418 | 66.7% | -0.1295 ± 0.0394 | 100.0% | 위와 같음 |
| 개인 | persist | 0.0867 ± 0.0324 | 100.0% | -0.0161 ± 0.0210 | 82.2% | 위와 같음 |

- 가중치 표의 전체 경로: `runs/report_observability_vkospi_e2/reward_weights_summary.csv`, `runs/report_observability_vkospi_e2/context_weights_summary.csv`
- 검증: 각 weight history의 200에폭 값과 최종 가중치 파일을 대조했으며 최대 절대 오차는 0
- 해석: 외국인과 개인의 핵심 `beta` 방향은 5~20에폭 사이에 대부분 형성된다. 개인 underwater는 초기 양수에서 10에폭에 음수로 전환되므로, 최종값만 제시하는 것보다 학습 경로를 함께 보여주는 편이 타당하다. 기관은 NLL 감소폭이 작아 최종 가중치 해석에 주의가 필요하다.
- 다음 액션: 보고서에는 전체 `beta` 변화표와 3행 NLL 수렴표를 본문에 넣고, `B` 변화표는 부록 또는 보조표로 배치

## 근거 구분

- 결론 근거: `runs/report_observability_vkospi_e2/`의 실제 재실행 결과
- Notion/GitHub 메모만으로 판단한 항목: 없음
- 코드·설정 변경만으로 추론한 항목: 없음
