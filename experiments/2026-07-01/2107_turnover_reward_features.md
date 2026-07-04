# 거래대금 보상특징 추가 실험

- 날짜/시간: 2026-07-01 21:07 KST
- 목적: Notion의 거래대금 보상특징을 참고해 20·60·120일 거래대금 활발도를 추가하고 기존 투자자별 champion과 비교
- 변경한 것: `log(trading_value)`의 rolling z-score를 계산하고, 이진 행동에서 식별 가능한 `action × turnover_z` 형태의 `turnover_20`, `turnover_60`, `turnover_120`을 추가
- 고정 조건: 기존 E2 `kospi_return_1d` 컨텍스트, 기존 5개 피처, 동일한 973개 날짜·라벨, CPCV 10-fold/2-test-fold(45 split), purge 1, embargo 5, seed 42, 투자자별 champion epoch·batch·learning rate·L1
- 데이터: `data/processed/dataset_vkospi_turnover.npz`
- 설정 파일: `experiments/2026-07-01/configs/features_vkospi_turnover.yaml`, `experiments/2026-07-01/configs/turnover_investor_specific.yaml`
- 결과 폴더: `runs/turnover_investor_specific/`
- 주요 결과: 세 투자자 모두 Accuracy와 Macro F1이 하락했고 NLL도 개선되지 않았다. 현재 champion에는 거래대금 3개 피처를 포함하지 않는다.

## 설계 근거와 변경 이유

- Notion 원문: [보상 특징](https://app.notion.com/p/36d20f0320248074a77ae9fe1d363a68)
- Notion 설계 노트: [거래대금 (Trading Volume) 설계 노트](https://app.notion.com/p/36e20f03202480d9b198f304d168fb4e)
- 원문의 기본식 `|a| × turnover_z`는 현재 행동집합 `{-1,+1}`에서 두 행동 모두 `|a|=1`이다. 따라서 두 logit에 같은 값이 더해져 softmax에서 완전히 상쇄되고 가중치를 식별할 수 없다.
- 설계 노트의 방향 구분 확장을 이진 행동에 맞게 `a × turnover_z`로 축약했다. 이는 매수와 매도 반응의 차이를 window당 하나의 계수로 식별한다.
- rolling z-score는 현재와 과거 거래대금만 사용한다. 기존 데이터셋과 새 데이터셋의 날짜·라벨이 정확히 같은 것도 확인했다.

## 성능 결과

| 투자자 | 기존 Accuracy | 거래대금 Accuracy | 변화 | 기존 Macro F1 | 거래대금 Macro F1 | 변화 | 기존 NLL | 거래대금 NLL | 변화 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 외국인 | 0.6379 | 0.6297 | -0.0082 | 0.6220 | 0.6142 | -0.0078 | 0.6498 | 0.6499 | +0.0001 |
| 기관 | 0.5457 | 0.5146 | -0.0311 | 0.5356 | 0.5099 | -0.0257 | 0.6923 | 0.6938 | +0.0015 |
| 개인 | 0.6008 | 0.5765 | -0.0244 | 0.5881 | 0.5633 | -0.0248 | 0.6784 | 0.6809 | +0.0024 |

| 투자자 | Accuracy 개선 split | Macro F1 개선 split | NLL 개선 split |
| --- | ---: | ---: | ---: |
| 외국인 | 12/45 (26.7%) | 15/45 (33.3%) | 24/45 (53.3%) |
| 기관 | 10/45 (22.2%) | 12/45 (26.7%) | 11/45 (24.4%) |
| 개인 | 5/45 (11.1%) | 6/45 (13.3%) | 12/45 (26.7%) |

- 기존 출처: `runs/hyperparameter_tuned_investor_specific/cv_metrics.csv`
- 거래대금 출처: `runs/turnover_investor_specific/cv_metrics.csv`

## 가중치 결과

| 대상 | 피처/가중치 | 값 | 변동성/일관성 | 출처 |
| --- | --- | ---: | ---: | --- |
| 외국인 | turnover_20 beta | -0.027606 | std 0.021195 / 음 84.4% | `runs/turnover_investor_specific/reward_weights_summary.csv` |
| 외국인 | turnover_60 beta | 0.050668 | std 0.017279 / 양 100.0% | 위와 같음 |
| 외국인 | turnover_120 beta | -0.018889 | std 0.026320 / 음 75.6% | 위와 같음 |
| 기관 | turnover_20 beta | 0.003366 | std 0.003359 / 양 86.7% | 위와 같음 |
| 기관 | turnover_60 beta | 0.002111 | std 0.004995 / 양 64.4% | 위와 같음 |
| 기관 | turnover_120 beta | 0.002742 | std 0.004989 / 양 71.1% | 위와 같음 |
| 개인 | turnover_20 beta | -0.002015 | std 0.010338 / 음 60.0% | 위와 같음 |
| 개인 | turnover_60 beta | -0.013077 | std 0.007174 / 음 97.8% | 위와 같음 |
| 개인 | turnover_120 beta | -0.003270 | std 0.010112 / 음 68.9% | 위와 같음 |
| 외국인 | turnover_20 × KOSPI B | 0.077493 | std 0.023613 / 양 100.0% | `runs/turnover_investor_specific/context_weights_summary.csv` |
| 외국인 | turnover_60 × KOSPI B | 0.006201 | std 0.011431 / 양 71.1% | 위와 같음 |
| 외국인 | turnover_120 × KOSPI B | -0.026845 | std 0.021825 / 음 91.1% | 위와 같음 |
| 기관 | turnover_20 × KOSPI B | 0.006446 | std 0.001979 / 양 100.0% | 위와 같음 |
| 기관 | turnover_60 × KOSPI B | 0.000445 | std 0.003597 / 양 46.7% | 위와 같음 |
| 기관 | turnover_120 × KOSPI B | -0.001684 | std 0.004169 / 음 77.8% | 위와 같음 |
| 개인 | turnover_20 × KOSPI B | -0.029822 | std 0.002893 / 음 100.0% | 위와 같음 |
| 개인 | turnover_60 × KOSPI B | -0.014679 | std 0.006974 / 음 97.8% | 위와 같음 |
| 개인 | turnover_120 × KOSPI B | -0.012740 | std 0.007405 / 음 93.3% | 위와 같음 |

- 전체 기존·신규 피처 가중치: `runs/turnover_investor_specific/reward_weights_summary.csv`
- 해석: 일부 거래대금 계수의 방향은 안정적이지만, 45개 outer test split의 예측 성능은 일관되게 나빠졌다. 방향 일관성만으로 유용한 보상특징이라고 결론 내리지 않는다.
- 다음 액션: 현재 모델에는 미포함. 추가 검토가 필요하면 세 window를 동시에 넣지 않고 `turnover_20/60/120` 각각의 단일-feature ablation으로 분리해야 한다.

## 검증

- 거래대금 피처 부호·과거값 전용·기존 피처 테스트: 통과
- 새 데이터셋과 기존 데이터셋: 날짜 및 라벨 완전 일치
- 거래대금 피처: 973개 표본 모두 유한값, 매수/매도 반대칭 오차 0

## 근거 구분

- 결론 근거: `runs/turnover_investor_specific/`와 기존 champion의 실제 실행 파일
- Notion 근거: 거래대금 정의와 20·60·120일 window 후보
- 코드·설정 변경만으로 추론한 항목: `|a|` 식의 이진 softmax 비식별성
