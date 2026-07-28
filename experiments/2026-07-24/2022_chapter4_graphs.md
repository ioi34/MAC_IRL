# 4장 그래프 초안 4종

- 날짜/시간: 2026-07-24 20:22
- 목적: 논문 4장에 사용할 후보 그래프 4종을 기존 연속형 주모형 결과로 작성한다.
- 변경한 것: 새 학습 없이 주모형·비교모형 CSV를 읽어 그래프와 재현용 파생 CSV를 생성했다.
- 고정 조건: 연속형 모형, 보상 특징 3개(momentum, herd, underwater), 컨텍스트 KOSPI200 1일 수익률·USD/KRW 252일 수준, κ=1, seed 42, CPCV 45개 분할.
- 데이터: 삼성전자 973거래일, 2022-01-06–2025-12-29.
- 설정 파일: `runs/continuous_reward3_ctxmain_default/config_snapshot.yaml`
- 결과 폴더: `experiments/2026-07-24/2022_chapter4_graphs/`
- 주요 결과:
  - `01_context_effective_weights`: 컨텍스트 수준에 따른 `β+B·C`의 변화를 투자자·특징별로 표시했다.
  - `02_actual_vs_prediction_rolling20d`: 날짜별 9개 CPCV 시험예측을 평균한 뒤 실제·예측 행동의 20거래일 이동평균을 비교했다.
  - `03_beta_alpha_forest`: 45개 분할의 `β`, `α` 분포와 평균±표준편차를 표시했다.
  - `04_paired_ablation_delta`: 같은 CPCV 분할에서 주모형과 컨텍스트 없음·상호작용 `B`만 모형의 성능 차이를 표시했다.
  - 이번 작업은 시각화만 수행했으며 새로운 실험 결과나 가중치를 생성하지 않았다.

## 가중치 결과

아래 값은 이번 그래프가 사용한 기존 주모형의 실제 45개 분할 요약값이다.

| 대상 | 피처/가중치 | 평균 | 표준편차 | 부호 일관성 | 출처 |
| --- | --- | ---: | ---: | ---: | --- |
| 외국인 | β momentum | 0.049555 | 0.008904 | 양 100.0% | `runs/continuous_reward3_ctxmain_default/reward_weights_summary.csv` |
| 외국인 | β herd | -0.040186 | 0.006583 | 음 100.0% | `runs/continuous_reward3_ctxmain_default/reward_weights_summary.csv` |
| 외국인 | β underwater | 0.009432 | 0.006204 | 양 95.6% | `runs/continuous_reward3_ctxmain_default/reward_weights_summary.csv` |
| 기관 | β momentum | -0.000593 | 0.001181 | 음 71.1% | `runs/continuous_reward3_ctxmain_default/reward_weights_summary.csv` |
| 기관 | β herd | -0.004380 | 0.000552 | 음 100.0% | `runs/continuous_reward3_ctxmain_default/reward_weights_summary.csv` |
| 기관 | β underwater | -0.000846 | 0.000910 | 음 80.0% | `runs/continuous_reward3_ctxmain_default/reward_weights_summary.csv` |
| 개인 | β momentum | -0.030354 | 0.001467 | 음 100.0% | `runs/continuous_reward3_ctxmain_default/reward_weights_summary.csv` |
| 개인 | β herd | -0.032187 | 0.001207 | 음 100.0% | `runs/continuous_reward3_ctxmain_default/reward_weights_summary.csv` |
| 개인 | β underwater | 0.029287 | 0.002871 | 양 100.0% | `runs/continuous_reward3_ctxmain_default/reward_weights_summary.csv` |
| 외국인 | α KOSPI200 1일 수익률 | 0.036143 | 0.006325 | 양 100.0% | `runs/continuous_reward3_ctxmain_default/context_main_weights_summary.csv` |
| 외국인 | α USD/KRW 252일 수준 | -0.026122 | 0.010768 | 음 100.0% | `runs/continuous_reward3_ctxmain_default/context_main_weights_summary.csv` |
| 기관 | α KOSPI200 1일 수익률 | -0.003232 | 0.001263 | 음 97.8% | `runs/continuous_reward3_ctxmain_default/context_main_weights_summary.csv` |
| 기관 | α USD/KRW 252일 수준 | -0.001802 | 0.001546 | 음 95.6% | `runs/continuous_reward3_ctxmain_default/context_main_weights_summary.csv` |
| 개인 | α KOSPI200 1일 수익률 | -0.024633 | 0.003464 | 음 100.0% | `runs/continuous_reward3_ctxmain_default/context_main_weights_summary.csv` |
| 개인 | α USD/KRW 252일 수준 | 0.029803 | 0.003223 | 양 100.0% | `runs/continuous_reward3_ctxmain_default/context_main_weights_summary.csv` |

- 상호작용 `B`의 전체 실제 분할값과 요약값은 각각 `runs/continuous_reward3_ctxmain_default/context_weights.csv`, `runs/continuous_reward3_ctxmain_default/context_weights_summary.csv`를 사용했다.
- 해석: 그래프는 기존 표의 수치를 대체 확정하는 것이 아니라, 투자자 이질성·분할 간 변동성·컨텍스트 조건부 민감도를 시각적으로 점검하기 위한 초안이다.
- 다음 액션: 네 그래프 중 본문에 넣을 대상을 선택한 뒤 논문 폭과 캡션 형식에 맞춰 크기·레이블을 최종 조정한다.

