# 최종 보상특징·컨텍스트 확정

- 날짜/시간: 2026-07-04 16:49 KST
- 목적: 신경망 구조 전 논문 주모형의 해석 가능한 보상특징과 컨텍스트 확정
- 변경한 것: 최종 보상특징을 `underwater + herd(5) + KRW momentum(20) + Parkinson volatility(20)`, 컨텍스트를 `KOSPI200 return(1d) + FX level z-score(252d)`로 승격. persist·직접 환율변화 보상특징·달러환산 모멘텀 제외
- 고정 조건: binary 행동, 973표본, CPCV 10-fold/2-test-fold 45 split, purge 1, embargo 5, seed 42, 200 epochs, batch 256, learning rate 0.01, L1 0.001
- 데이터: `samsung_macirl_EXTENDED_2019_2025.csv`, 2022-01-06~2025-12-29
- 설정 파일: `configs/data_context.yaml`, `configs/features_final.yaml`, `configs/model.yaml`, `configs/train.yaml`, `configs/experiment_final.yaml`
- 결과 폴더: `runs/final_reward_context/`
- 주요 결과: 환율 수준이 없는 KOSPI-only 기준선 대비 전체 Macro F1 0.570044 → 0.572644, NLL 0.682159 → 0.682609. 환율 수준 컨텍스트는 외국인의 underwater·herd·momentum 반응과 개인의 underwater·herd·momentum 반응에서 안정적인 방향을 보였다.

## 최종 구성

| 구분 | 변수 | 역할 |
| --- | --- | --- |
| 보상특징 | underwater | 평균단가 대비 손실구간의 매수·매도 선호 |
| 보상특징 | herd(5) | 타 투자자 최근 흐름 추종 |
| 보상특징 | KRW momentum(20) | 원화 기준 추세추종·역추세 |
| 보상특징 | Parkinson volatility(20) | 고변동성에서의 매수·매도 방향 반응 |
| 컨텍스트 | KOSPI200 return(1d) | 단기 시장 방향에 따른 보상계수 변화 |
| 컨텍스트 | FX level z-score(252d) | 과거 1년 대비 환율 수준에 따른 보상계수 변화 |

## 성능 결과

| 투자자 | Accuracy | Macro F1 | NLL |
| --- | ---: | ---: | ---: |
| 외국인 | 0.631753 | 0.615702 | 0.664711 |
| 기관 | 0.533771 | 0.523756 | 0.702855 |
| 개인 | 0.592464 | 0.578475 | 0.680260 |
| 전체 평균 | 0.585996 | 0.572644 | 0.682609 |

## 기본 보상 가중치

| 투자자 | 피처 | 평균 | 표준편차 | 부호 일관성 | 출처 |
| --- | --- | ---: | ---: | ---: | --- |
| 외국인 | underwater | -0.057869 | 0.038067 | 91.1% 음 | `runs/final_reward_context/reward_weights_summary.csv` |
| 외국인 | herd | -0.231670 | 0.047269 | 100.0% 음 | 위와 같음 |
| 외국인 | momentum | 0.139340 | 0.037065 | 100.0% 양 | 위와 같음 |
| 외국인 | volatility | 0.006814 | 0.057271 | 53.3% 양 | 위와 같음 |
| 기관 | underwater | -0.006162 | 0.022866 | 62.2% 음 | 위와 같음 |
| 기관 | herd | -0.047678 | 0.022139 | 100.0% 음 | 위와 같음 |
| 기관 | momentum | -0.060825 | 0.030399 | 97.8% 음 | 위와 같음 |
| 기관 | volatility | 0.041390 | 0.024262 | 93.3% 양 | 위와 같음 |
| 개인 | underwater | 0.103063 | 0.042121 | 100.0% 양 | 위와 같음 |
| 개인 | herd | -0.168692 | 0.036292 | 100.0% 음 | 위와 같음 |
| 개인 | momentum | -0.058475 | 0.030514 | 97.8% 음 | 위와 같음 |
| 개인 | volatility | -0.048813 | 0.042008 | 88.9% 음 | 위와 같음 |

## 주요 컨텍스트 가중치

| 투자자 | 피처 × 컨텍스트 | 평균 | 표준편차 | 부호 일관성 | 출처 |
| --- | --- | ---: | ---: | ---: | --- |
| 외국인 | underwater × FX level | -0.080886 | 0.041081 | 100.0% 음 | `runs/final_reward_context/context_weights_summary.csv` |
| 외국인 | herd × FX level | -0.051830 | 0.046680 | 82.2% 음 | 위와 같음 |
| 외국인 | momentum × FX level | -0.163610 | 0.056006 | 100.0% 음 | 위와 같음 |
| 외국인 | momentum × KOSPI | -0.128644 | 0.028036 | 100.0% 음 | 위와 같음 |
| 기관 | underwater × KOSPI | 0.062631 | 0.026121 | 100.0% 양 | 위와 같음 |
| 기관 | herd × FX level | -0.057050 | 0.025080 | 95.6% 음 | 위와 같음 |
| 기관 | volatility × KOSPI | -0.070529 | 0.024056 | 100.0% 음 | 위와 같음 |
| 기관 | volatility × FX level | 0.035435 | 0.020004 | 97.8% 양 | 위와 같음 |
| 개인 | underwater × FX level | -0.116892 | 0.039930 | 100.0% 음 | 위와 같음 |
| 개인 | herd × FX level | -0.061426 | 0.036549 | 91.1% 음 | 위와 같음 |
| 개인 | momentum × FX level | 0.122974 | 0.037969 | 100.0% 양 | 위와 같음 |
| 개인 | volatility × KOSPI | -0.111055 | 0.038100 | 100.0% 음 | 위와 같음 |

- 해석: 외국인은 고환율 국면에서 underwater·herd·momentum의 유효 가중치가 더 음(-)의 방향으로 이동한다. 이는 환율의 직접 매수·매도 효과가 아니라 환율 수준이 기존 행동 메커니즘을 조절하는 조건부 패턴이다.
- 비채택: 직접 FX 변화 보상특징은 기관에서만 안정적이었고, 달러환산 모멘텀은 외국인 계수 해석은 안정적이지만 원화 모멘텀보다 성능이 소폭 낮았다. 두 결과는 부록/강건성 분석으로 유지한다.
- 다음 액션: 이 선형 contextual reward를 해석 기준선으로 고정하고 신경망은 비선형 강건성 비교에만 사용

## 검증 및 근거 구분

- 실제 파일 기반: `runs/final_reward_context/`의 45 split × 3투자자 결과
- 재현 확인: 최종 기본 명령 결과가 선택 실험 `runs/fx_semantics_e1_level_context/`의 성능·가중치와 일치
- 테스트: 전체 39개 통과, 기존 `herd_pairwise` registry 기대값 불일치 1개 실패
- Notion/GitHub 메모만으로 판단한 항목: 없음
- 코드·설정에서 추론한 항목: 없음
