# merged_e2 전체 보상특징 Ablation

- 날짜/시간: 2026-07-06 00:23 KST
- 목적: GitHub PR #22에서 확정되고 PR #23에서 유지된 `merged_e2`의 6개 보상특징이 투자자별 행동 예측에 실질적으로 기여하는지 검증
- 변경한 것: 기준선 재학습, 6개 특징 개별 제거, 행동재무 그룹과 전통 특징 그룹 제거
- 고정 조건: `kospi_return_1d + fx_level_z_252` 컨텍스트, 투자자별 `merged_e2` HP, 973표본, CPCV 10-fold/2-test-fold 45 split, purge 1, embargo 5, seed 42
- 데이터: `samsung_macirl_EXTENDED_V2019_2025.csv`, 2022-01-06~2025-12-29
- 설정 파일: `experiments/2026-07-06/configs/merged_e2_ablation/`
- 결과 폴더: `runs/ablation_merged_e2/`
- 주요 결과: 개별 제거에서 외국인의 `momentum`은 Accuracy와 Macro F1, 개인의 `momentum`은 NLL을 유의하게 악화시켰다. 나머지 특징의 개별 제거 효과는 다중검정 보정 후 유의하지 않았다.

## 기준선 성능

| 투자자 | Accuracy | Macro F1 | NLL |
| --- | ---: | ---: | ---: |
| 외국인 | 0.613911 | 0.596005 | 0.668744 |
| 기관 | 0.513233 | 0.494329 | 0.692712 |
| 개인 | 0.586694 | 0.575034 | 0.675937 |

기준선 결과는 GitHub의 `merged_e2` 기록과 일치한다.

## 개별 제거 결과

양수 변화량은 특징 제거로 성능이 악화됐다는 뜻이다. OOF 변화량은 반복 test 예측확률을 날짜별로 평균한 뒤 20거래일 moving-block bootstrap 10,000회로 검정했다. `q`는 투자자·지표별 6개 특징에 Benjamini-Hochberg 보정을 적용한 값이다.

| 제거 특징 | 투자자 | 유의 지표 | OOF 악화폭 | 95% CI | q | 판정 |
| --- | --- | --- | ---: | --- | ---: | --- |
| momentum | 외국인 | Accuracy | 0.023638 | [0.006166, 0.042138] | 0.029397 | 유의 |
| momentum | 외국인 | Macro F1 | 0.023686 | [0.005856, 0.042061] | 0.025797 | 유의 |
| momentum | 개인 | NLL | 0.005402 | [0.001040, 0.010327] | 0.035996 | 유의 |
| underwater | 전체 | 없음 | - | - | - | 유의한 기여 미확인 |
| relative | 전체 | 없음 | - | - | - | 단독 기여 미확인 |
| volatility | 전체 | 없음 | - | - | - | 유의한 기여 미확인 |
| herd_a | 전체 | 없음 | - | - | - | 유의한 기여 미확인 |
| herd_b | 전체 | 없음 | - | - | - | 유의한 기여 미확인 |

- 전체 변화량·신뢰구간·split 승률: `runs/ablation_merged_e2/metric_comparison.csv`
- 각 모델 성능: `runs/ablation_merged_e2/model_metrics.csv`

## 그룹 제거와 공선성

| 제거 그룹 | 투자자 | Accuracy 악화 | Macro F1 악화 | NLL 악화 | 해석 |
| --- | --- | ---: | ---: | ---: | --- |
| 행동재무(`underwater+herd_a+herd_b`) | 외국인 | 0.000000 | -0.001432 | -0.006449 | 유의한 악화 없음 |
| 행동재무 | 기관 | 0.000000 | -0.002667 | 0.000145 | 유의한 악화 없음 |
| 행동재무 | 개인 | 0.010277 | 0.010670 | 0.003260 | CI가 0을 포함 |
| 전통(`momentum+relative`) | 외국인 | 0.040082 | 0.041085 | 0.022561 | Accuracy·NLL CI가 0 초과, F1은 경계 |
| 전통 | 기관 | -0.006166 | 0.005254 | -0.000348 | 유의한 악화 없음 |
| 전통 | 개인 | 0.040082 | 0.040738 | 0.011126 | 세 지표 CI가 모두 0 초과 |

모든 VIF는 5 미만이다. 최대값은 기관 `herd_a=3.270376`, `herd_b=3.130073`이다. 다만 기관의 `herd_a-herd_b` 상관은 -0.819510으로 높고, `momentum-relative` 상관은 모든 투자자에서 0.673170이다.

- VIF: `runs/ablation_merged_e2/feature_vif.csv`
- 상관계수: `runs/ablation_merged_e2/feature_correlations.csv`

## 가중치 결과

### 기준선 β

| 투자자 | 피처 | 평균 | 표준편차 | 부호 일관성 | 출처 |
| --- | --- | ---: | ---: | ---: | --- |
| 외국인 | underwater | -0.048614 | 0.026596 | 97.8% 음 | `runs/ablation_merged_e2/baseline/reward_weights_summary.csv` |
| 외국인 | momentum | 0.118265 | 0.011951 | 100.0% 양 | 위와 같음 |
| 외국인 | relative | 0.117733 | 0.022672 | 100.0% 양 | 위와 같음 |
| 외국인 | volatility | 0.022726 | 0.042811 | 68.9% 양 | 위와 같음 |
| 외국인 | herd_a | -0.030812 | 0.021876 | 86.7% 음 | 위와 같음 |
| 외국인 | herd_b | -0.102401 | 0.014372 | 100.0% 음 | 위와 같음 |
| 기관 | underwater | -0.002767 | 0.002323 | 80.0% 음 | 위와 같음 |
| 기관 | momentum | -0.004911 | 0.000046 | 100.0% 음 | 위와 같음 |
| 기관 | relative | -0.002591 | 0.003098 | 77.8% 음 | 위와 같음 |
| 기관 | volatility | 0.004322 | 0.001356 | 93.3% 양 | 위와 같음 |
| 기관 | herd_a | -0.000972 | 0.002695 | 68.9% 음 | 위와 같음 |
| 기관 | herd_b | -0.003038 | 0.002532 | 82.2% 음 | 위와 같음 |
| 개인 | underwater | 0.028212 | 0.009224 | 97.8% 양 | 위와 같음 |
| 개인 | momentum | -0.035053 | 0.001347 | 100.0% 음 | 위와 같음 |
| 개인 | relative | -0.035888 | 0.000817 | 100.0% 음 | 위와 같음 |
| 개인 | volatility | -0.016170 | 0.015652 | 82.2% 음 | 위와 같음 |
| 개인 | herd_a | -0.033790 | 0.002361 | 100.0% 음 | 위와 같음 |
| 개인 | herd_b | -0.033629 | 0.003345 | 100.0% 음 | 위와 같음 |

### 제거 모델의 실제 β 평균

| 모델 | 투자자 | 남은 피처의 β 평균 |
| --- | --- | --- |
| remove_underwater | 외국인 | momentum=0.120945, relative=0.115591, volatility=0.005280, herd_a=-0.025159, herd_b=-0.105885 |
| remove_underwater | 기관 | momentum=-0.004906, relative=-0.002568, volatility=0.004277, herd_a=-0.000914, herd_b=-0.003064 |
| remove_underwater | 개인 | momentum=-0.035434, relative=-0.036366, volatility=-0.014119, herd_a=-0.034235, herd_b=-0.033766 |
| remove_momentum | 외국인 | underwater=-0.060965, relative=0.144956, volatility=0.043830, herd_a=-0.033988, herd_b=-0.120939 |
| remove_momentum | 기관 | underwater=-0.002768, relative=-0.002645, volatility=0.004358, herd_a=-0.001112, herd_b=-0.002957 |
| remove_momentum | 개인 | underwater=0.028770, relative=-0.036885, volatility=-0.018545, herd_a=-0.034943, herd_b=-0.033643 |
| remove_relative | 외국인 | underwater=-0.050949, momentum=0.144700, volatility=0.022610, herd_a=-0.034537, herd_b=-0.118502 |
| remove_relative | 기관 | underwater=-0.002746, momentum=-0.004917, volatility=0.004361, herd_a=-0.001060, herd_b=-0.002994 |
| remove_relative | 개인 | underwater=0.029701, momentum=-0.036224, volatility=-0.015116, herd_a=-0.034770, herd_b=-0.033737 |
| remove_volatility | 외국인 | underwater=-0.042934, momentum=0.131661, relative=0.115219, herd_a=-0.034335, herd_b=-0.107204 |
| remove_volatility | 기관 | underwater=-0.002602, momentum=-0.004921, relative=-0.002583, herd_a=-0.001071, herd_b=-0.003010 |
| remove_volatility | 개인 | underwater=0.028525, momentum=-0.035657, relative=-0.035912, herd_a=-0.033999, herd_b=-0.033488 |
| remove_herd_a | 외국인 | underwater=-0.047013, momentum=0.118287, relative=0.118268, volatility=0.023719, herd_b=-0.095459 |
| remove_herd_a | 기관 | underwater=-0.002747, momentum=-0.004905, relative=-0.002558, volatility=0.004353, herd_b=-0.002973 |
| remove_herd_a | 개인 | underwater=0.029171, momentum=-0.035782, relative=-0.036437, volatility=-0.016621, herd_b=-0.033438 |
| remove_herd_b | 외국인 | underwater=-0.058120, momentum=0.130419, relative=0.129175, volatility=0.028592, herd_a=0.010173 |
| remove_herd_b | 기관 | underwater=-0.002780, momentum=-0.004896, relative=-0.002542, volatility=0.004351, herd_a=-0.000647 |
| remove_herd_b | 개인 | underwater=0.028456, momentum=-0.035121, relative=-0.035938, volatility=-0.015785, herd_a=-0.033672 |
| remove_behavioral_group | 외국인 | momentum=0.134110, relative=0.128487, volatility=0.006235 |
| remove_behavioral_group | 기관 | momentum=-0.004880, relative=-0.002480, volatility=0.004326 |
| remove_behavioral_group | 개인 | momentum=-0.036212, relative=-0.036946, volatility=-0.014165 |
| remove_traditional_group | 외국인 | underwater=-0.067609, volatility=0.040012, herd_a=-0.038770, herd_b=-0.143759 |
| remove_traditional_group | 기관 | underwater=-0.002717, volatility=0.004386, herd_a=-0.001206, herd_b=-0.002893 |
| remove_traditional_group | 개인 | underwater=0.030171, volatility=-0.017631, herd_a=-0.035893, herd_b=-0.033743 |

전체 β 표준편차·부호 일관성은 `runs/ablation_merged_e2/reward_weights_summary.csv`, 컨텍스트 가중치는 `runs/ablation_merged_e2/context_weights_summary.csv`에 있다.

- 해석:
  - 외국인에서 momentum 제거 시 relative가 0.117733→0.144956, relative 제거 시 momentum이 0.118265→0.144700으로 커져 일부 대체가 확인된다. 그럼에도 momentum 단독 제거는 유의한 성능 하락을 남겼다.
  - 개인은 momentum과 relative 단독 제거의 Accuracy/F1 효과가 각각 유의하지 않지만, 둘을 함께 제거하면 세 지표가 모두 악화했다. 두 특징을 개별적으로 “장식”이라고 판단하면 안 된다.
  - 행동재무 그룹은 세 투자자 모두 통계적으로 유의한 성능 악화를 만들지 못했다. 현재 결과만으로 underwater와 pairwise herd의 예측 기여를 강하게 주장하기 어렵다.
  - 기관은 어떤 개별·그룹 제거에서도 안정적인 성능 하락이 확인되지 않았다.
- 다음 액션: momentum은 유지 근거가 가장 강하다. relative는 momentum과의 결합 기여로 유지 여부를 판단하고, underwater·volatility·herd_a·herd_b는 가중치 안정성 및 경제적 해석과 함께 별도 근거가 필요하다.

## 검증 및 근거 구분

- 실제 파일 기반: `runs/ablation_merged_e2/`의 기준선과 8개 제거 모델, 총 1,215개 독립 투자자 fit
- 테스트: 44개 통과
- 데이터 일치: 모든 모델 973개 날짜, 동일 라벨·컨텍스트 사용
- GitHub 근거: PR #22의 `merged_e2` 확정 구성, PR #23에서 특징 구성 유지
- Notion/GitHub 메모만으로 판단한 결론: 없음
- 코드·설정만으로 추론한 결론: 없음
