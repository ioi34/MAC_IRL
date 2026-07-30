# CPCV와 walk-forward의 역할 구분

> **대상 실험**: 3특징 확정 사양 — `momentum` · `persist` · `underwater`,
> 컨텍스트 `kospi_return_1d` · `fx_level_z_252`, 컨텍스트 **주효과 α 사용**, κ=1, 연속 정책.
> 학습 `runs/continuous_reward3_persist/`, 검증 `runs/continuous_reward3_persist_validation/`
>
> 이 문서는 "왜 두 가지 분할을 다 쓰는가"에 답한다. 코드에서 직접 확인한 내용만 적는다.
> 관련: `docs/persist_spec_and_results.md`, `experiments/2026-07-28/2038_persist_검증스위트.md`

---

## 0. 한 줄 요약

**CPCV는 추정의 정밀도를, walk-forward는 시간에 대한 견고성을 답한다.**
서로 대체할 수 없고, 각 주장은 정확히 한쪽에만 귀속시켜야 한다.

---

## 1. 검증 스위트의 네 항목은 서로 독립이다

`scripts/run_continuous_reward_validation.py`의 `main()` 실행 순서 (코드 확인):

1. 변형 8개 각각에 `_run_cpcv(...)` → **ablation = CPCV**
2. `_run_walk_forward(baseline_feature_path, ...)` → **baseline 하나에만**
3. ridge 강도 15개 각각에 `_run_cpcv(...)` → **ridge = CPCV**
4. `_run_monthly_block_bootstrap(...)`

| 검증 항목 | 데이터 분할 | 적용 대상 | 산출물 |
| --- | --- | --- | --- |
| canonical 학습 | CPCV 45 split | 확정 사양 | β·α 평균±표준편차, OOS 성능 |
| ablation | CPCV 45 split | 변형 8개 | 변형별 풀링 OOS + paired bootstrap |
| ridge 대조 | CPCV 45 split | 강도 15개 | 부호 일치 여부 |
| walk-forward | 확장창 시계열 | **baseline만** | 연도별 β, 연도별 OOS |
| 월블록 부트스트랩 | **분할 없음 (in-sample)** | baseline만 | β의 95% 구간 |

> 부트스트랩은 리샘플된 전체 데이터에 in-sample로 학습한다. 테스트 분할이 없어
> `predictions.csv`·`cv_metrics.csv`가 생성되지 않는다. **계수 안정성 전용 도구**이며
> 성능 수치를 여기서 인용해서는 안 된다.

---

## 2. CPCV — "추정치가 얼마나 확실한가"

### 2.1 구조

`src/data/splits.py`는 skfolio의 `CombinatorialPurgedCV`를 사용한다.
설정(`configs/train.yaml`): `n_folds: 10`, `n_test_folds: 2`, `purged_size: 1`, `embargo_size: 5`.

973일을 시간순 10개 폴드(각 약 97일)로 자르고, 그중 **2개를 테스트로 고르는 모든 조합**
C(10,2) = **45개** split을 만든다. 각 split은 나머지 8폴드에서 purge·embargo를 뺀 구간으로
학습하고, 고른 2폴드에서 평가한다.

실측 크기 (`runs/continuous_reward3_persist/cv_splits.csv`, 45행):

| 항목 | 값 |
| --- | --- |
| train | 766 ~ 773일 |
| test | 194일 (2폴드) |
| purge + embargo로 제외 | 6 ~ 13일 |

- **purge 1**: 테스트 경계에 인접한 관측을 학습에서 제거 → 라벨 겹침 방지
- **embargo 5**: 테스트 직후 5일을 학습에서 차단 → 자기상관을 통한 누출 방지
- `train_continuous.py` L260에 train/test 교집합 검사가 있어 겹치면 즉시 예외 발생

### 2.2 핵심 성질 — 테스트 폴드는 연속이 아니다

테스트로 고른 2폴드는 인접할 필요가 없다. **폴드 0과 7을 테스트로 쓰면서 1~6, 8~9로
학습하는 split이 존재한다.** 즉 테스트 시점보다 미래의 데이터로 학습하는 경우가 섞인다.

이는 버그가 아니라 CPCV의 설계 의도다(López de Prado). 목적이 실거래 시뮬레이션이 아니라,
**짧은 단일 시계열에서 서로 다른 train/test 분할을 최대한 많이 뽑아내는 것**이기 때문이다.

### 2.3 우리 실험에서 CPCV가 담당하는 것 — 세 가지

**(1) β·α의 표본분포**

45개 독립 추정치가 나오므로 평균·표준편차·부호 일관성을 계산할 수 있다.
`persist +0.0505 ± 0.0060, 부호 100%`라고 쓸 수 있는 근거는 **오직 이것**이다.
단일 홀드아웃이면 값 하나만 나오고 ±도 %도 없다.

**(2) OOS 성능의 산포**

각 split의 테스트 예측은 학습에 쓰이지 않은 구간이므로 out-of-sample이다.
`방향정확도 0.6435 ± 0.0395`의 ±는 45개 분할에 걸친 산포다.

**(3) ablation·ridge 비교의 공통 기반**

변형 8개와 ridge 15개가 **동일한 45 split**에서 실행되므로, 차이가 분할 운이 아니라
사양 차이라고 말할 수 있다. paired block bootstrap은 그 위에서 날짜를 짝지어 검정한다.

### 2.4 CPCV가 답하지 못하는 것

- **시간적 일반화.** 미래로 학습하는 split이 섞여 있으므로 "2023년에 실시간으로
  돌렸다면 됐을까"에 답할 수 없다. 성능이 낙관적으로 편향될 여지가 있다.
- **레짐 변화.** 45개 분할이 모두 전 기간에 걸쳐 있어 특정 시기의 붕괴가 평균에 묻힌다.
- **계수가 0과 구분되는지.** CPCV의 표준편차는 *분할 간 산포*이지 추정오차가 아니다.
  이 질문은 부트스트랩 95% 구간이 답한다.

---

## 3. Walk-forward — "시간이 지나도 유지되는가"

### 3.1 구조

`scripts/train_continuous_walk_forward.py` (코드 확인):

```python
train_indices = np.flatnonzero(dates.year < test_year)
test_indices  = np.flatnonzero(dates.year == test_year)
if purge:
    train_indices = train_indices[:-purge]   # 학습 마지막 1일 제거
```

확장창(expanding window), 엄격한 시간순, **미래 누출 없음.**

| 테스트 연도 | 학습 | 테스트 |
| --- | ---: | ---: |
| 2023 | 242일 (2022년만) | 245일 |
| 2024 | 487일 | 244일 |
| 2025 | 731일 | 241일 |

스케일러도 각 윈도우의 학습 구간에서만 적합(`fit_context_scaler(contexts, train_indices)`).

### 3.2 우리 실험에서 walk-forward가 담당하는 것 — 두 가지뿐

**(1) 계수 부호의 시간 안정성** — CPCV가 구조적으로 못 하는 검사

| 대상 | momentum | persist | underwater |
| --- | :---: | :---: | :---: |
| 외국인 | 무반전 ✔ | **무반전 ✔** (std 0.0021) | 반전 ✘ |
| 기관 | 반전 ✘ | 반전 ✘ | 반전 ✘ |
| 개인 | 무반전 ✔ | **무반전 ✔** | 반전 ✘ |

외국인 persist의 표준편차 0.0021은 9개 계수 중 최소(min +0.0508, max +0.0549).
출처: `.../analysis/walk_forward_reward_stability.csv`

**(2) 레짐 의존성 진단**

2023년 개인 붕괴(방향정확도 0.376, 상관 −0.198)는 **walk-forward에서만 보인다.**
CPCV 평균(개인 0.6120 / 0.2478)은 이를 지워버린다.
출처: `.../analysis/walk_forward_metrics.csv`

### 3.3 walk-forward의 한계 — 세 가지 모두 실재

1. **윈도우 3개뿐** → 유의성 검정 불가. 부호·크기 비교만 가능.
2. **baseline만 실행** → `remove_persist` 대응 실행이 없으므로 **persist의 기여에 대해
   아무 말도 할 수 없다.** persist 기여 검정은 ablation(CPCV) 단독 근거다.
3. **2023 윈도우는 학습이 242일뿐** → 그 실패가 레짐 때문인지 표본 부족 때문인지
   분리되지 않는다.

---

## 4. 주장별 근거 귀속 표

논문 작성 시 이 표를 기준으로 인용한다. 오른쪽 열의 것을 근거로 대면 오류다.

| 주장 | 올바른 근거 | 근거가 **아닌** 것 |
| --- | --- | --- |
| persist β = +0.0505, 부호 일관성 100% | CPCV 45 split | — |
| 외국인 방향정확도 0.6435 ± 0.0395 | CPCV 45 split | — |
| persist를 빼면 유의하게 나빠진다 | **ablation (CPCV) 단독** | walk-forward (미실행) |
| 계수가 0과 구분된다 (95% CI) | 월블록 부트스트랩 | CPCV 표준편차 |
| 3년간 부호가 유지된다 | **walk-forward 단독** | CPCV (미래 학습 포함) |
| 2023년 개인 레짐 취약 | **walk-forward 단독** | CPCV (평균에 묻힘) |
| 결론이 L1 선택에 의존하지 않는다 | ridge 그리드 (CPCV) | — |
| 기관 momentum은 잡음이다 | 부트스트랩 CI + ablation + walk-forward 반전 (3중) | — |

---

## 5. 집계 방식 차이 주의

같은 CPCV 결과라도 집계가 다르면 수치가 달라진다.

| 집계 | 방식 | 외국인 상관 |
| --- | --- | ---: |
| split별 평균 (§4.1, canonical) | 45개 split에서 각각 상관 계산 → 평균 | **0.306** |
| 풀링 (ablation 표) | 45개 split의 OOS 예측을 날짜별로 합쳐 상관 1회 계산 | **0.344** |

ablation이 풀링을 쓰는 이유는 paired bootstrap이 baseline과 변형을 **같은 날짜에 짝지어**
비교해야 하기 때문이다. 둘 다 CPCV OOS이고 집계만 다르다.

→ **논문 본문 성능표는 split별 평균(0.306)을 쓰고, ablation 표에는 집계 방식이 다르다는
각주를 단다.** 두 표의 숫자를 직접 비교하면 안 된다.

---

## 6. 논문에 반영할 서술 방향

**심사자 반론 대비가 walk-forward를 넣은 실질적 이유다.** "CPCV는 테스트 시점보다 미래
데이터로 학습하므로 성능이 낙관적으로 편향된다"는 지적이 나올 수 있고, walk-forward가
그에 대한 답이다.

단 이 답은 **"계수 부호는 견딘다"까지만** 하고, 예측 성능은 레짐 의존적이라고 인정하는
형태가 정직하다. 2023년 개인의 붕괴를 숨기지 않는다.

권장 서술 골격:

> 계수 추정의 정밀도는 CPCV 45 split과 월블록 부트스트랩으로, 시간적 견고성은 확장창
> walk-forward로 평가한다. CPCV는 짧은 표본에서 다수의 분할을 확보해 추정치의 산포를
> 주지만, 테스트 시점 이후 데이터가 학습에 포함될 수 있어 시간적 일반화를 보장하지 않는다.
> walk-forward는 이를 보완하며, 외국인·개인의 momentum과 persist 계수가 3개 연도에서
> 부호 반전 없이 유지됨을 확인했다. 다만 2023년 개인 표본에서는 예측 성능이 크게
> 저하되어(방향정확도 0.376), 구조 해석은 견디나 예측 성능은 레짐 의존적임을 밝힌다.

---

## 7. 남은 결정 사항

1. **walk-forward에 ablation을 추가할 것인가** — 변형별로 `_run_walk_forward`를 호출하도록
   스크립트 수정이 필요하다. 윈도우 3개라 유의성 검정은 불가하고 부호·크기 비교만 가능하므로,
   8배 실행 비용 대비 얻는 것이 적다. **현 상태 유지 + 논문에서 근거를 정확히 귀속**하는
   쪽을 권장. 사용자 판단 필요.
2. 그룹 ablation 2종(`remove_behavioral_group`·`remove_traditional_group`)은 3특징에서
   단일 제거와 특징셋이 동일한 축퇴 → 논문 표에서 제외.
