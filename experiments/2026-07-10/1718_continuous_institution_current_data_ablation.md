# 현재 데이터 기반 기관 연속행동 특징·컨텍스트 ablation

- 날짜/시간: 2026-07-10 17:18 KST
- 목적: 외부 데이터를 추가하지 않고 기관 순매수 강도의 설명력과 가중치 해석력을 높일 보상특징·컨텍스트를 선별한다.
- 변경한 것: 기관 주문 지속성, 1·5일 시장조정 수익률, 20일 벤치마크 drift, 분기말 proximity, VKOSPI 변화, 유동성 capacity를 구현했다. 컨텍스트는 사전 지정한 feature-context 쌍만 학습하는 masked B를 적용했다. 기관만 학습하는 실행 옵션과 공통 평가·walk-forward 분석을 추가했다.
- 고정 조건: 973표본(2022-01-06~2025-12-29), 동일 연속 타깃, CPCV 10-fold/2-test-fold 45 split, purge 1, embargo 5, 종가 t까지의 정보로 t+1 행동 예측, 기관 설정 epoch 10·full batch 2048·lr 0.0005·L1 0.01.
- 데이터: `samsung_macirl_EXTENDED_V2019_2025.csv`, `data/processed/dataset_continuous_institution_candidates.npz`
- 설정 파일: `experiments/2026-07-10/configs/continuous_inst_e0_baseline.yaml` ~ `continuous_inst_e7_persistence_liquidity.yaml`
- 결과 폴더: `runs/continuous_institution_ablation/`, `runs/continuous_institution_seed_robustness/`, `runs/continuous_institution_walk_forward/`
- 주요 결과: `execution_persistence_3`만 명확하게 채택 기준을 통과했다. CPCV 45개 split 모두에서 기준선보다 RMSE가 낮고, 방향·상관·큰 행동 적중률이 함께 개선됐다. 유동성×지속성은 추가 개선폭이 너무 작고 연도별 우위가 일관되지 않아 본문 핵심이 아니라 부록 후보로 남긴다.

## 데이터 버전 확인

- VKOSPI 파일과 기존 연속 파일은 날짜·가격·투자자 거래열이 같고, `usdkrw` 841행과 `kospi200_return` 4행이 다르다.
- VKOSPI를 쓰는 후보와 공정하게 비교하기 위해 기준선 E0도 VKOSPI 파일로 다시 실행했다.
- 행동 배열·973개 날짜·행동 요약 통계는 기존 연속 데이터와 일치한다.

## CPCV 공통 평가 결과

| 실험 | 추가 가설 | RMSE | RMSE skill | Balanced Acc. | Macro F1 | Pearson | 예측/실제 σ | 상위 25% 방향정확도 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E0 | 기준선 | 0.124611 | 0.005060 | 0.517181 | 0.506616 | 0.077149 | 0.069445 | 0.568444 |
| E1 | 주문 지속성 | **0.123962** | **0.010136** | 0.540617 | 0.534020 | 0.131007 | 0.087123 | 0.624481 |
| E2 | 1일 시장조정 수익률 | 0.124628 | 0.004919 | 0.510038 | 0.499502 | 0.075617 | 0.072254 | 0.568961 |
| E3 | 5일 시장조정 수익률 | 0.124644 | 0.004785 | 0.513575 | 0.503065 | 0.073673 | 0.069901 | 0.564825 |
| E4 | 20일 벤치마크 drift | 0.124684 | 0.004457 | 0.511767 | 0.501060 | 0.068019 | 0.070916 | 0.559810 |
| E5 | drift×분기말 | 0.124711 | 0.004223 | 0.512107 | 0.501514 | 0.065431 | 0.071816 | 0.560753 |
| E6 | 지속성×VKOSPI | 0.124011 | 0.009763 | 0.537611 | 0.530871 | 0.125758 | 0.087905 | 0.624063 |
| E7 | 지속성×유동성 | **0.123956** | **0.010185** | **0.543197** | **0.537059** | **0.132001** | **0.088678** | **0.625370** |

- E1 vs E0: RMSE -0.000648, 45/45 split 승리, Balanced Accuracy +0.0234, Pearson +0.0539, 상위 행동 방향정확도 +0.0560.
- E7 vs E1: RMSE -0.000006, 31/45 split 승리. 수치는 개선됐지만 효과 크기는 사실상 미미하다.
- E2는 28/45 split에서 RMSE가 낮았지만 평균 RMSE가 악화됐고 방향지표도 하락해 탈락했다.
- 출처: `experiments/2026-07-10/institution_current_data_analysis/institution_ablation_summary.csv`, `institution_paired_comparison.csv`.

## 가중치 결과

| 실험 | 대상 | 피처/가중치 | 값 | 변동성/일관성 | 출처 |
| --- | --- | --- | ---: | ---: | --- |
| E1 | institution | beta execution_persistence_3 | 0.004773 | std 0.000062 / 양 100.0% | `runs/continuous_institution_ablation/e1_persistence/reward_weights_summary.csv` |
| E2 | institution | beta short_residual_return_1 | 0.001571 | std 0.001372 / 양 95.6% | `runs/continuous_institution_ablation/e2_residual_1d/reward_weights_summary.csv` |
| E3 | institution | beta short_residual_return_5 | 0.000048 | std 0.000646 / 양 51.1% | `runs/continuous_institution_ablation/e3_residual_5d/reward_weights_summary.csv` |
| E4 | institution | beta benchmark_drift_20 | 0.000307 | std 0.000978 / 양 51.1% | `runs/continuous_institution_ablation/e4_benchmark_drift/reward_weights_summary.csv` |
| E5 | institution | B drift×quarter_end | 0.000415 | std 0.000697 / 양 68.9% | `runs/continuous_institution_ablation/e5_drift_quarter_end/context_weights_summary.csv` |
| E6 | institution | B persistence×VKOSPI return | 0.001088 | std 0.001279 / 양 75.6% | `runs/continuous_institution_ablation/e6_persistence_vkospi/context_weights_summary.csv` |
| E7 | institution | beta execution_persistence_3 | 0.004770 | std 0.000062 / 양 100.0% | `runs/continuous_institution_ablation/e7_persistence_liquidity/reward_weights_summary.csv` |
| E7 | institution | B persistence×liquidity capacity | 0.001011 | std 0.001091 / 양 91.1% | `runs/continuous_institution_ablation/e7_persistence_liquidity/context_weights_summary.csv` |

- E1 해석: 오늘까지 같은 방향으로 누적 집행한 기관은 다음 거래일에도 그 방향의 순매수 불균형을 이어가는 경향이 있다. 주문 분할·집행 지속 가설과 일치한다.
- E2는 사전 가설인 일별 contrarian(음수)과 반대로 양수지만 성능도 개선하지 못해 채택하지 않는다.
- E5는 사전 가설 B<0과 반대이며 일관성 68.9%, E6도 일관성 75.6%로 80% 기준 미달이다.
- E7 유동성 B는 양수로, 유동성이 좋은 국면에서 주문 지속성이 조금 강해진다는 해석이 가능하지만 std가 평균보다 크고 성능 효과가 매우 작다.

## expanding walk-forward

| 테스트 연도 | E0 RMSE / skill | E1 RMSE / skill | E7 RMSE / skill | E1 Pearson | E7 Pearson |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2023 | 0.119465 / -0.009289 | **0.118637 / -0.002290** | 0.118641 / -0.002328 | -0.162040 | -0.161409 |
| 2024 | 0.126104 / -0.011838 | 0.125684 / -0.008470 | **0.125618 / -0.007940** | 0.071498 | 0.079046 |
| 2025 | 0.127025 / 0.020543 | 0.126129 / 0.027447 | **0.126108 / 0.027612** | 0.163137 | 0.164804 |

- E1은 세 연도 모두 E0보다 RMSE가 낮고 Pearson과 상위 25% 행동 방향정확도도 모두 개선했다.
- 2023·2024 skill은 여전히 음수라 평균 예측을 안정적으로 이겼다고 할 수 없다.
- E7은 E1 대비 2023에 미세 악화, 2024·2025에 미세 개선했다. 본문 주특징으로 추가할 실익이 부족하다.
- 출처: 각 `runs/continuous_institution_walk_forward/*/walk_forward_metrics.csv`.

## seed 및 검증

- seed 42~46 결과는 1e-10 수준까지 동일했다. 기관 batch 2048이 모든 train split보다 커서 full-batch이고 초기 파라미터도 0으로 고정되어 현재 학습이 결정론적이기 때문이다. seed 강건성 증거로 해석하지 않는다.
- 전체 테스트: 52개 통과.
- `git diff --check`: 통과.

## 최종 판단

- **본문 채택:** `execution_persistence_3`.
- **부록/강건성:** `execution_persistence_3 × liquidity_capacity_20`.
- **탈락:** 1일·5일 residual return, benchmark drift, drift×quarter-end, persistence×VKOSPI.
- 현재 최선 E1도 CPCV RMSE skill 1.01%, NRMSE 1.001 수준이라 기관 행동 설명 문제가 해결된 것은 아니다. 다음 단계는 E1을 고정한 뒤 연속모형 전용 epoch·L1 튜닝을 nested train 구간 안에서 수행하는 것이다.

## 근거 구분

- 실제 파일 기반: 모든 성능·가중치·walk-forward·seed 결과.
- 논문/기존 가설 기반: 주문 분할 지속성, contrarian, 벤치마크 리밸런싱, 유동성 제약의 예상 부호.
- 코드·설정에서 추론한 항목: full-batch 때문에 seed 결과가 동일한 원인.
- 외부 데이터 기반 후보: 미실행 및 본 실험에서 제외.
