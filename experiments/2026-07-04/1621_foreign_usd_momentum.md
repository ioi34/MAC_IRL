# 외국인 달러환산 모멘텀 대체 실험

> 최종 주모형에서는 비채택했다. 최종 구성은 `experiments/2026-07-04/1649_final_reward_context_selection.md`를 따른다.

- 날짜/시간: 2026-07-04 16:21 KST
- 목적: 외국인 환율 반응을 주가와 분리된 환율계수 대신 달러 기준 투자성과로 해석할 수 있는지 검증
- 변경한 것: `momentum_20 = log(P_t/P_{t-20})`을 `usd_momentum_20 = log(P_t/P_{t-20}) - log(USDKRW_t/USDKRW_{t-20})`으로 완전 대체
- 고정 조건: underwater, herd(5), Parkinson volatility(20), KOSPI200 1일 컨텍스트, persist·직접 환율민감도 제외, binary 행동, 동일 973일·라벨·CPCV 45 split, seed 42, 200 epochs, batch 256, learning rate 0.01, L1 0.001
- 데이터: `samsung_macirl_EXTENDED_2019_2025.csv`, 2022-01-06~2025-12-29
- 설정 파일: `experiments/2026-07-04/configs/features_foreign_usd_momentum.yaml`, `experiments/2026-07-04/configs/foreign_usd_momentum.yaml`
- 결과 폴더: `runs/foreign_usd_momentum/`
- 주요 결과: 외국인 달러환산 모멘텀 계수는 +0.099518, 45개 split에서 100% 양수였다. 연도별 단순 행동 상관도 2022~2025 모두 양수였다. 그러나 원화 모멘텀 기준선보다 Accuracy 0.003459, Macro F1 0.003480 하락하고 NLL 0.003612 악화했다.

## 외국인 성능 결과

| 모형 | Accuracy | Macro F1 | NLL | Accuracy 개선 split | F1 개선 split | NLL 개선 split |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 원화 모멘텀 | **0.639101** | **0.621962** | **0.654749** | - | - | - |
| 달러환산 모멘텀 | 0.635642 | 0.618481 | 0.658361 | 40.0% | 42.2% | 15.6% |

## 가중치 결과

| 대상 | 피처/가중치 | 값 | 표준편차/부호 일관성 | 출처 |
| --- | --- | ---: | ---: | --- |
| 외국인 기준선 | momentum beta | 0.135025 | 0.034589 / 100.0% 양 | `runs/fx_semantics_e0_no_fx/reward_weights_summary.csv` |
| 외국인 실험 | usd_momentum beta | 0.099518 | 0.037404 / 100.0% 양 | `runs/foreign_usd_momentum/reward_weights_summary.csv` |
| 외국인 실험 | underwater beta | -0.088634 | 0.038191 / 100.0% 음 | 위와 같음 |
| 외국인 실험 | herd beta | -0.252158 | 0.045529 / 100.0% 음 | 위와 같음 |
| 외국인 실험 | volatility beta | 0.040824 | 0.056309 / 75.6% 양 | 위와 같음 |
| 외국인 실험 | usd_momentum × KOSPI B | -0.075337 | 0.028291 / 97.8% 음 | `runs/foreign_usd_momentum/context_weights_summary.csv` |

## 기간별 방향

| 연도 | USD momentum과 외국인 이진 행동 상관 | 방향 |
| --- | ---: | --- |
| 2022 | 0.1758 | 양 |
| 2023 | 0.2312 | 양 |
| 2024 | 0.2921 | 양 |
| 2025 | 0.1027 | 양 |

- 해석: 외국인은 달러 기준 20일 성과가 높을수록 다음 날 매수 방향을 선택하는 positive-feedback 패턴을 보였다. 계수 부호와 연도별 방향은 매우 안정적이다.
- 채택 판단: 예측 champion으로는 원화 모멘텀을 유지한다. 해석 우선 외국인 사양에서는 달러환산 모멘텀을 채택할 수 있다. 사전 guardrail인 NLL 악화 0.005 이내, Macro F1 악화 0.01 이내를 충족하며 환율 경로를 직접 해석할 수 있기 때문이다.
- 주의: 현재 실험은 동일 파이프라인 때문에 국내 투자자에도 달러 모멘텀을 계산했지만 채택 판단은 외국인 결과에만 적용한다. 외국인 집단 전체를 달러 투자자로 근사한다는 한계가 있다.
- 다음 액션: 외국인만 달러 기준, 기관·개인은 원화 기준을 사용하도록 투자자 기준통화 모멘텀을 구성하거나, 달러 기준 underwater를 추가 검증

## 검증 및 근거 구분

- 실제 파일 기반: `runs/foreign_usd_momentum/`, 기준선 `runs/fx_semantics_e0_no_fx/`
- 데이터 검증: 두 모형의 날짜, 라벨, underwater, herd, volatility, KOSPI 컨텍스트가 정확히 동일
- 테스트: 전체 40개 통과, 기존 `herd_pairwise` registry 불일치 1개 실패
- Notion/GitHub 메모만으로 판단한 항목: 없음
- 코드·설정에서 추론한 항목: 달러환산 모멘텀의 경제적 의미
