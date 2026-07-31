# 정확 Lasso(coordinate descent) 대조 — Adam 해가 전역해인지 검정

- 날짜/시간: 2026-07-31 13:29
- 목적: 리뷰어 단점 2. 원고 §3.5는 목적함수가 볼록이고 유일해를 가짐을
  rank/등상관집합으로 검증해놓고 정작 추정은 Adam으로 한다. 게다가 §3.5 첫 문장은
  "coordinate descent attains a global solution"이라 써놓고 뒤에서 Adam을 쓴다고 해
  **원고 내부에 모순이 있다.** 동일 split·동일 목적함수를 정확 솔버로 다시 풀어
  두 해가 일치하는지 확인한다.
- 변경한 것: **없음.** 학습·데이터·설정 일체 불변. 기존 canonical의 계수 CSV를 읽어
  대조만 했다.
- 고정 조건: 3특징(momentum·persist·underwater), 맥락 2개, κ=1, λ=0.005 통일,
  CPCV 10/2/purge1/embargo5 = 45 split, split별 train-only 표준화, n=973.
- 데이터: `data/processed/dataset_continuous_reward3_persist.npz`
- 대조 대상 run: `runs/continuous_reward3_persist_epochs75/`
- 스크립트: `scripts/compare_exact_lasso.py`
- 결과 폴더: `experiments/2026-07-31/exact_lasso/`

## 목적함수 대응

| | 목적함수 | |
| --- | --- | --- |
| 원고 (`continuous_trainer.py:106-108`) | `(1/n) Σ (θᵀw − y)² + λ‖θ‖₁` | λ = 0.005 |
| sklearn `Lasso` | `(1/2n)‖y − Xθ‖² + α‖θ‖₁` | |

원고 = 2 × sklearn 이므로 **α = λ/2 = 0.0025**. `fit_intercept=False`
(원고에 절편 없음). L1은 11개 파라미터 전부에 걸린다
(`continuous.py:107-113` — β 3 + B 6 + α 2).

설계행렬은 `continuous.py`의 `forward`를 그대로 옮겼다:
`w = [x(3) ; rvec(x cᵀ)(6) ; c(2)]`.

## 주요 결과

1. **포화율이 정확히 0이다** (45 split × 3유형 전부). 클리핑이 한 번도 결속되지
   않으므로 §3.5의 "reduction is exact rather than approximate" 주장이
   **처음으로 실측 근거를 얻었다.** 출처 `exact_lasso/saturation.csv`.
2. **해석 대상 5계수의 CPCV 평균이 전부 0.001 이내로 일치한다.** 최대 차이는
   개인 underwater의 0.00099. 헤드라인(외국인 momentum +0.044, 개인 −0.036)은
   소수 넷째 자리까지 동일하다.
3. **식별 결론이 하나도 바뀌지 않는다.** 외국인·개인의 momentum·persist 및
   외국인 KOSPI, 개인 KOSPI·FX는 정확 솔버에서도 45/45 부호 일치.
   외국인 FX는 오히려 97.8% → 100%로 개선.
4. **기관 널은 더 강해진다.** 정확 솔버는 기관 momentum을 45 split 중 **55.6%**,
   기관 underwater를 **77.8%**에서 *정확히 0*으로 만든다. Adam은 L1 하위경사에서
   미소 비영값을 남기므로 "불안정"으로 보였을 뿐, 정확해에서는 **아예 선택되지
   않는 변수**다. 널 결과의 서술이 "unstable"에서 "not selected"로 강해진다.
5. **부호 불일치 265/1485건은 대부분 허수다.** 그중 253건이 `exact=0` vs
   `adam=미소값` 조합이다. 양쪽 모두 |값|>1e-4인 진짜 불일치는 **11건**뿐이고,
   전부 상호작용항이거나 이미 미식별로 보고된 계수다.
6. ⚠️ **단, split 단위 차이는 무시할 수 없다.** 외국인 KOSPI에서 최대 0.0167
   (계수 크기 0.033의 절반)까지 벌어진다. 이는 2026-07-30 수렴 진단에서 외국인이
   가장 덜 수렴했던 것(θ 마지막 변화 3.93%, 다른 둘은 1.7%/1.2%)과 정확히 일치한다.
   **평균은 맞지만 split별 Adam 해는 전역해가 아니다.**

## 가중치 결과

해석 대상 5계수, 45 split 기준. 출처
`experiments/2026-07-31/exact_lasso/{interpreted_coefficients,coefficient_comparison}.csv`

| 대상 | 피처/가중치 | 정확 Lasso 평균 | Adam 평균 | 평균차 | exact=0 인 split 비율 | 최대 split 차이 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 외국인 | momentum | +0.04373 | +0.04354 | +0.00018 | 0.0% | 0.0089 |
| 외국인 | persist | +0.05104 | +0.05055 | +0.00049 | 0.0% | 0.0147 |
| 외국인 | underwater | +0.00655 | +0.00612 | +0.00043 | 6.7% | 0.0127 |
| 외국인 | kospi_return_1d | +0.03293 | +0.03310 | −0.00017 | 0.0% | **0.0167** |
| 외국인 | fx_level_z_252 | −0.02225 | −0.02150 | −0.00075 | 0.0% | 0.0103 |
| 기관 | momentum | −0.00146 | −0.00148 | +0.00002 | **55.6%** | 0.0004 |
| 기관 | persist | +0.01031 | +0.01024 | +0.00007 | 0.0% | 0.0002 |
| 기관 | underwater | −0.00021 | −0.00028 | +0.00007 | **77.8%** | 0.0005 |
| 기관 | kospi_return_1d | −0.00594 | −0.00597 | +0.00003 | 0.0% | 0.0002 |
| 기관 | fx_level_z_252 | −0.00347 | −0.00349 | +0.00002 | 13.3% | 0.0004 |
| 개인 | momentum | −0.03605 | −0.03556 | −0.00049 | 0.0% | 0.0051 |
| 개인 | persist | +0.04668 | +0.04620 | +0.00048 | 0.0% | 0.0032 |
| 개인 | underwater | +0.02784 | +0.02883 | −0.00099 | 8.9% | 0.0097 |
| 개인 | kospi_return_1d | −0.02187 | −0.02157 | −0.00030 | 0.0% | 0.0029 |
| 개인 | fx_level_z_252 | +0.03321 | +0.03284 | +0.00036 | 0.0% | 0.0032 |

### V1 부호 일관성 대조 (sign(0)=0 으로 계산)

| 대상 | 피처 | Adam | 정확 Lasso |
| --- | --- | ---: | ---: |
| 외국인 | momentum / persist / kospi | 100% | 100% |
| 외국인 | fx | 97.8% | **100%** |
| 외국인 | underwater | 91.1% | 93.3% |
| 기관 | persist / kospi | 100% | 100% |
| 기관 | momentum | 75.6% | **44.4%** (0으로 선택 제외) |
| 기관 | underwater | 75.6% | **22.2%** (〃) |
| 기관 | fx | 97.8% | 86.7% |
| 개인 | momentum / persist / kospi / fx | 100% | 100% |
| 개인 | underwater | 97.8% | 91.1% |

## 해석

리뷰어 지적은 타당했다. Adam은 split 단위로 전역해에 도달하지 못했다(외국인 최대
0.0167). 그러나 **CPCV 평균과 모든 식별 결론은 정확 솔버에서 그대로 재현된다.**
따라서 논문의 결론은 안전하고, 정확 솔버 결과를 함께 보고하면 리뷰어의 우려를
원천 차단하면서 기관 널 서술도 강해진다.

## 다음 액션

1. **§3.5 개정.** "Problem (5) is convex and coordinate descent attains a global
   solution" 뒤에 실측 문장 추가. 예:
   > An exact coordinate-descent solve of~(5) on the same 45 splits reproduces the
   > CPCV mean of every interpreted coefficient to within $0.001$ in standardized
   > units and leaves every identification verdict unchanged. The saturation rate
   > is exactly zero in all 135 fits, so the reduction in~(5) is exact rather than
   > approximate on this sample.
2. **§4.2 수렴 문단(3줄) 삭제 가능.** "The 75-epoch fits also make incomplete
   optimization an unlikely explanation…" 문단은 정확 솔버 결과로 대체된다.
   페이지도 5줄 확보된다(2026-07-31 페이지 측정 H 변형).
3. **기관 널 서술 강화 검토.** §4.3의 "Momentum and loss-region weights are near
   zero and unstable"을 "정확 솔버는 45 split 중 각각 55.6%·77.8%에서 이들을 0으로
   선택 제외한다"로 바꾸면 훨씬 강한 진술이 된다. **단 §4는 동료 작업 중이므로 조율 필요.**
4. 정직하게 남길 것: split 단위 편차(최대 0.0167, 외국인)는 숨기지 말고
   "Adam이 split별 전역해에 도달하지 못했으므로 정확 솔버 결과를 병기한다"로 서술.
