# persist 명세 확정 및 실행 결과

> **확정: `persist`(자기분모)** — 3특징 = `momentum` · `persist` · `underwater`.
> 기존 `herd`를 대체. 근거·비교 실험·torch 정식 실행 절차를 정리한다.
> 관련: `docs/herd_sias2002_analysis.md`, `docs/herd_persist_decomposition_plan.md`,
> `experiments/2026-07-28/herd_persist_교체.md`

---

## 1. 교체 배경 (요약)

기존 `herd`는 시장청산 항등식 때문에 **자기시차의 부호 반전**이 되어 있었다.

세 유형이 시장을 거의 소진하므로 $\sum_j u_j \approx 0$ (실측: $\sum u$ 평균절대값 0.0133
vs $|u_{\text{foreign}}|$ 평균 0.1254, 잔차 약 10%). 따라서

$$\phi^{\text{herd}}_{i,t} = \frac{1}{2}\Big(\sum_j u_{j,t-1} - u_{i,t-1}\Big) \approx -\frac{1}{2}\,u_{i,t-1}$$

실측 상관: 외국인 **−0.989**, 기관 −0.974, 개인 −0.992.

→ herd는 "다른 유형의 flow"가 아니라 **자기 전일 flow**를 측정하고 있었고, 부호 일관성
100%도 행동 증거가 아니라 회계 항등식의 결과였다. `persist`는 같은 정보를 직접·양의
부호로 측정하므로 이름과 측정이 일치한다.

---

## 2. 확정 명세 — `persist`

`src/features/persist.py`

$$\phi^{\text{persist}}_{i,t} = a_{i,t-1} \times a, \qquad a_{i,t} = \frac{\text{buy\_value}_{i,t} - \text{sell\_value}_{i,t}}{\text{buy\_value}_{i,t} + \text{sell\_value}_{i,t}}$$

```python
def build_persist(df, config, investor, action):
    return investor_trade_imbalance(df, config, investor).shift(1) * action
```

**어제의 행동 그 자체.** 모델이 예측하는 행동 $a$와 **동일한 정의**(자기 총거래 분모)를
사용하므로 계수가 행동에 대한 문자 그대로의 AR(1)이 된다.

### 자기분모를 택한 이유 (공통분모 명세는 폐기)

초기 구현은 공통분모($u = $ 순매수/종목전체거래대금)를 사용했으나 **특징과 행동이 서로 다른
척도**였다. "어제의 시장점유율 기준 순매수"로 "오늘의 자기거래 기준 순매수"를 예측하는
구조라 "자기 행동의 지속"이라는 해석이 엄밀히 성립하지 않는다.

> 예: 어제 거래량은 적었으나 대부분 매수였다면 자기분모로는 $a \approx +0.8$(강한 매수
> 성향)이지만 공통분모로는 $u \approx +0.01$(시장에서 미미). 공통분모 버전은
> "성향의 지속"이 아니라 "시장 영향력의 지속"을 측정한다.
>
> **공통분모 구현은 코드·config에서 제거했다.** `src/features/persist.py`는 자기분모
> 단일 구현이며, 레지스트리 키도 `persist` 하나다.

이 선택은 결과에 맞춘 조정이 아니라 **정의상 정합성 수정**이다.

### 검토했으나 기각한 대안

| 대안 | 기각 사유 |
| --- | --- |
| herd 유지 + 이름만 변경 | 측정 실패(자기시차)가 해소되지 않음 |
| herd를 max/\|·\|max로 변경 | 효과 있는 버전은 의도와 불일치(max는 "더 매수한 쪽"), 의도에 맞는 버전은 개선 없음, 개선이 정보 손실에서 옴, 문헌 근거 없음 |
| momentum에 직교화 | 공선성 문제 없음(VIF<1.5)에도 변수를 손보는 것이라 데이터 마이닝 우려 |
| persist + herd (Sias 분해 재현) | 시장청산으로 두 성분이 기계적 결합(−0.99) → 분리 추정 불가 |

---

## 3. 수식 정합성 확인

### $a$를 두 번 곱하는가 → 아니다

$$\phi^{\text{persist}}_{i,t} = \underbrace{a_{i,t-1}}_{\text{어제, 관측된 상수}} \times \underbrace{a}_{\text{오늘, 결정변수}}$$

$a_{i,t-1}$은 이미 실현된 데이터이므로 최적화에서 **계수**로 취급된다. 보상은

$$R(a) = \big[\beta_p\,a_{i,t-1} + \beta_m \text{mom}_t + \beta_u \text{uw}_t\big]\,a - \tfrac12\kappa a^2$$

로 $a$에 대해 **1차항 + 2차항**이며, $a^2$은 concave 항 하나뿐이다. **concave 논리 유지.**

### 다른 특징과 척도가 동일한가 → 그렇다

모든 특징이 "(관측된 상태 신호) × (오늘의 행동)" 구조로 통일되어 있고, split별
표준화(평균 0·분산 1)를 거치므로 계수 비교가 유효하다.

| 특징 | 상태 신호 |
| --- | --- |
| momentum | $\log(P_t/P_{t-20})$ |
| underwater | $\max\!\big(0, (\bar c_{i,t}-\mathrm{VWAP}_t)/\bar c_{i,t}\big)$, $\bar c$ = ρ=0.98 감쇠 평균단가 |
| **persist** | $a_{i,t-1}$ |

단, 성격 구분은 논문에 명시한다: momentum·underwater는 **외부 상태 신호**, persist는
**자기 거래 이력**(Sias 분해의 자기추종 성분).

---

## 4. 실행 결과 (torch 정식 실행 — **확정**)

> **확정 수치**: `runs/continuous_reward3_persist/` (torch, 2026-07-28 실행).
> 아래 "persist(확정)" 열은 torch 값이다. 비교용 "canonical(herd)" 열은 numpy 재현이므로
> 열 간 비교는 **경향 확인용**이며, 논문 인용은 persist 열만 사용한다.
> numpy 예비 재현은 `runs/continuous_reward3_persist_numpy_prelim/`에 보존했고
> torch와 소수 셋째 자리까지 일치했다.

설정: canonical과 동일 — κ=1, 컨텍스트 주효과 α 사용, 컨텍스트 `kospi_return_1d`·
`fx_level_z_252`, CPCV 10/2/purge1/embargo5 = 45 split, split별 train-only 표준화,
투자자별 HP(외국인 ep75/lr0.001, 기관 ep10/bs2048/lr0.0005, 개인 ep20/bs512/lr0.001,
λ=0.005 통일), seed 42, 973행(2022-01-06~2025-12-29).

### 4.1 OOS 성능 (45 split 평균 ± 표준편차)

| 대상 | 지표 | canonical(herd) | **persist(확정)** |
| --- | --- | ---: | ---: |
| 외국인 | 방향정확도 | 0.6365 ± 0.0414 | **0.6435 ± 0.0395** |
| 외국인 | 상관 | 0.2837 ± 0.1056 | **0.3060 ± 0.1061** |
| 외국인 | RMSE | 0.2453 | **0.2423** |
| 기관 | 방향정확도 | 0.5332 ± 0.0376 | 0.5355 ± 0.0376 |
| 기관 | 상관 | 0.0731 ± 0.0596 | **0.0757 ± 0.0579** |
| 기관 | RMSE | 0.1250 | 0.1251 |
| 개인 | 방향정확도 | 0.6083 ± 0.0439 | **0.6120 ± 0.0458** |
| 개인 | 상관 | 0.2465 ± 0.1178 | **0.2478 ± 0.1205** |
| 개인 | RMSE | 0.3174 | **0.3172** |

MAE: 외국인 0.1917 / 기관 0.0945 / 개인 0.2638. 포화율 세 주체 모두 0.000.

→ persist가 전 주체에서 최고이거나 동등. 외국인 상관 **0.284 → 0.306**(+0.022).

### 4.2 β (평균 ± 표준편차, 부호 일관성)

| 대상 | 피처 | canonical(herd) | **persist(확정)** |
| --- | --- | ---: | ---: |
| 외국인 | momentum | +0.0476 ± 0.0089 (100%) | **+0.0435 ± 0.0090 (100%)** |
| 외국인 | herd → persist | −0.0382 ± 0.0065 (100%) | **+0.0505 ± 0.0060 (100%)** |
| 외국인 | underwater | +0.0050 ± 0.0048 (88.9%) | **+0.0061 ± 0.0048 (91.1%)** |
| 기관 | momentum | −0.0013 ± 0.0018 (62.2%) | **−0.0013 ± 0.0018 (71.1%)** |
| 기관 | herd → persist | −0.0047 ± 0.0002 (100%) | **+0.0047 ± 0.0003 (100%)** |
| 기관 | underwater | −0.0021 ± 0.0016 (93.3%) | **−0.0021 ± 0.0016 (93.3%)** |
| 개인 | momentum | −0.0301 ± 0.0015 (100%) | **−0.0301 ± 0.0015 (100%)** |
| 개인 | herd → persist | −0.0319 ± 0.0013 (100%) | **+0.0319 ± 0.0014 (100%)** |
| 개인 | underwater | +0.0287 ± 0.0031 (100%) | **+0.0289 ± 0.0031 (100%)** |

**momentum·underwater 계수가 사실상 불변** → 헤드라인(외국인 추세추종 / 개인 역행·
처분효과) 보존 확인. persist는 세 주체 모두 **양(+), 일관성 100%**.

### 4.3 α (컨텍스트 주효과)

| 대상 | kospi_return_1d | fx_level_z_252 |
| --- | ---: | ---: |
| 외국인 | +0.0331 (100%) | −0.0215 (97.8%) |
| 기관 | −0.0044 (100%) | −0.0033 (93.3%) |
| 개인 | −0.0235 (100%) | +0.0289 (100%) |

외국인·개인이 정확한 거울상. canonical과 동일 패턴.

### 4.4 공선성 진단

**특징 쌍별 공유 분산 (R²)**

| 쌍 | 외국인 | 기관 | 개인 |
| --- | ---: | ---: | ---: |
| momentum ↔ persist | 18.8% | 0.5% | 14.9% |
| momentum ↔ underwater | 4.5% | 0.6% | **34.3%** |
| persist ↔ underwater | 2.0% | 0.1% | 13.8% |

**각 특징의 "나머지 전부"에 대한 R² 및 VIF**

| 대상 | momentum | persist | underwater |
| --- | --- | --- | --- |
| 외국인 | 21.2% (VIF 1.27) | 19.1% (VIF 1.24) | 4.8% (VIF 1.05) |
| 기관 | 1.0% (VIF 1.01) | 0.6% (VIF 1.01) | 0.6% (VIF 1.01) |
| 개인 | **37.6%** (VIF 1.60) | 18.1% (VIF 1.22) | 36.8% (VIF 1.58) |

→ 전부 경고선(VIF 5) 대비 매우 낮음. **개인에서는 persist가 셋 중 가장 고유**하며,
기존 특징 쌍(momentum↔underwater 34.3%)이 오히려 더 얽혀 있다. underwater가 정의상
가격의 함수이므로 momentum과 겹치는 것은 자연스럽다.

### 4.5 저변동일 검증 — momentum으로 환원되지 않음

|일간수익| 하위 33%인 날(325/974일)만 골라 flow 자기상관을 재계산.

| 대상 | 전체 AR(1) | **저변동일 AR(1)** |
| --- | ---: | ---: |
| 외국인 | +0.394 | **+0.347** |
| 기관 | +0.156 | +0.113 |
| 개인 | +0.365 | **+0.375** |

**가격이 거의 움직이지 않은 날에도 지속성이 유지된다.** momentum으로는 설명 불가하며,
주문 분할 집행·거래 관성이라는 별도 메커니즘을 지지한다. **변수를 손대지 않고(직교화 없이)
데이터로 독립성을 입증**한 결과다.

### 4.6 행동의 AR(1) — 계수 해석의 직접 대응

자기분모 행동 $a_{i,t}$의 1차 자기상관: 외국인 **+0.403**, 기관 +0.151, 개인 **+0.344**.

persist는 이 행동 자체의 시차이므로, 추정 β(+0.051/+0.005/+0.032)의 부호·순서가
원자료 AR(1)과 일치한다. **계수 해석("어제 한 만큼 오늘도 한다")이 원자료 통계와 직접
대응**한다. 폐기된 공통분모 구현에서는 이 대응이 성립하지 않았다.

---

## 5. torch 정식 실행 + 검증 스위트 (**완료** · 2026-07-28)

§4는 아래 절차로 실행해 확정했다. numpy 재현과 소수 셋째 자리까지 일치.

### 5.1 준비된 파일

| 파일 | 내용 |
| --- | --- |
| `src/features/persist.py` | 특징 구현 (자기분모로 교체) |
| `src/features/registry.py` | `persist` 등록 (수정) |
| `configs/features_continuous_reward3_persist.yaml` | 3특징 명세 (신규) |
| `configs/experiment_continuous_reward3_persist.yaml` | canonical HP·컨텍스트 주효과 (신규) |
| `tests/test_feature_modules.py` | 레지스트리 기대 목록에 `persist` 추가 (수정) |

### 5.2 실행 명령

```bash
cd ~/Desktop/MAC_IRL

# 1) 전처리 (이미 생성돼 있으면 생략 가능)
python -m scripts.prepare_continuous_data \
  --data-config configs/data_continuous.yaml \
  --features-config configs/features_continuous_reward3_persist.yaml

# 2) canonical 학습 (45 CPCV split)
python -m scripts.train_continuous \
  --data-config configs/data_continuous.yaml \
  --features-config configs/features_continuous_reward3_persist.yaml \
  --model-config configs/model.yaml \
  --train-config configs/train.yaml \
  --experiment-config configs/experiment_continuous_reward3_persist.yaml

# 3) 검증 스위트 (bootstrap 200 / walk-forward / ridge / ablation)
python -m scripts.run_continuous_reward_validation \
  --data-config configs/data_continuous.yaml \
  --features-config configs/features_continuous_reward3_persist.yaml \
  --model-config configs/model.yaml \
  --train-config configs/train.yaml \
  --experiment-config configs/experiment_continuous_reward3_persist.yaml \
  --output-root runs/continuous_reward3_persist_validation \
  --bootstrap-resamples 200

# 4) 검증 결과 집계
python -m scripts.analyze_continuous_reward_validation \
  --run-root runs/continuous_reward3_persist_validation
```

산출 위치: `runs/continuous_reward3_persist/`(학습),
`runs/continuous_reward3_persist_validation/`(검증).

### 5.3 확인 결과

| 확인 항목 | 결과 |
| --- | :---: |
| momentum·underwater 계수 보존 (헤드라인 보호) | ✔ 외국인 +0.0435 / 개인 −0.0301 / 개인 uw +0.0289 |
| persist β 세 주체 모두 양(+)·일관성 100% | ✔ +0.0505 / +0.0047 / +0.0319 |
| numpy 예비와 소수 셋째 자리 일치 | ✔ |
| bootstrap 95% 구간이 0 배제 | ✔ persist만 3/3 |
| walk-forward 부호 반전 없음 | ◐ 외국인·개인 ✔, 기관 ✘ |
| ridge 부호 일치 | ✔ 9/9 |

### 5.4 검증 스위트 요약

전체 표·출처는 **`experiments/2026-07-28/2038_persist_검증스위트.md`** 참조.

**(1) 월블록 부트스트랩 (200 resample) — 95% 구간이 0을 배제하는가**

| 대상 | momentum | persist | underwater |
| --- | :---: | :---: | :---: |
| 외국인 | ✔ [+0.031, +0.075] | **✔ [+0.031, +0.072]** | ✘ [−0.018, +0.021] |
| 기관 | ✘ [−0.005, +0.004] | **✔ [+0.000, +0.005]** | ✘ [−0.005, +0.002] |
| 개인 | ✔ [−0.035, −0.028] | **✔ [+0.028, +0.036]** | ✔ [+0.026, +0.035] |

→ **persist는 세 주체 모두에서 0을 배제하는 유일한 특징.**
B(컨텍스트 상호작용)는 12개 중 2개만 0을 배제 → **B는 개별 계수 해석 금지, 패턴 수준만 서술.**

**(2) Walk-forward (2023/24/25) — 부호 반전**

외국인 persist 표준편차 **0.0021로 9개 계수 중 최소**(min +0.0508, max +0.0549).
외국인·개인은 momentum·persist 무반전, underwater는 반전. 기관은 세 특징 모두 반전.
2023년 개인 성능 붕괴(방향 0.376, 상관 −0.198)는 학습 242일 + 레짐 차이 → **한계로 명시.**

**(3) Ridge 대조 (강도: 외국인 1.0 / 기관 10.0 / 개인 0.3)**

부호 일치 **9/9 (100%)**, 성능 거의 동일. → 결론이 L1 선택에 의존하지 않는다.
개인은 ridge/lasso 절대비 0.99로 정규화에 사실상 무관.

**(4) Ablation (paired block bootstrap, block 20, 10,000 resample, BH 보정)**

*유의한 성능 저하는 전 변형·전 지표를 통틀어 `remove_persist` 외국인 2건뿐:*

| 변형 | 대상 | 지표 | Δ(저하) | 95% CI | q |
| --- | --- | --- | ---: | --- | ---: |
| remove_persist | 외국인 | 상관 | +0.0612 | [+0.032, +0.088] | **0.0014** |
| remove_persist | 외국인 | RMSE | +0.0060 | [+0.003, +0.009] | **0.0007** |

준유의(q≥0.05)도 전부 `remove_persist`: 기관 상관 +0.042, 개인 상관 +0.024, 개인 RMSE +0.0025.

*반대로 유의하게 **개선**되는 경우 (= 그 특징이 해로움):*

| 변형 | 대상 | 지표 | 변화 | q |
| --- | --- | --- | --- | ---: |
| remove_momentum | 기관 | 상관 | 0.061 → 0.099 | 0.0084 |
| remove_momentum | 기관 | RMSE | 0.12558 → 0.12518 | 0.0269 |
| remove_underwater | 개인 | 방향정확도 | 0.604 → 0.630 | 0.0025 |

> **주의 1** — ablation 표의 성능은 45 split 예측을 **풀링**한 값이라 §4.1의 split별
> 평균과 다르다(외국인 상관 풀링 0.344 vs split평균 0.306). 비교는 표 내부에서만 유효.
>
> **주의 2** — manifest의 `remove_behavioral_group`은 `remove_underwater`와,
> `remove_traditional_group`은 `remove_momentum`과 **특징셋이 동일**하다(5특징 시절 설계의
> 축퇴). 논문 표에서는 그룹 변형을 제외한다.

**(5) 공선성 재확인** — VIF 최대 1.60(개인 momentum). 개인에서 가장 얽힌 쌍은
momentum↔underwater(|r| 0.586)이고 **persist가 셋 중 가장 고유**(VIF 1.221).
"persist는 momentum의 재탕"이라는 반론은 데이터로 기각.

### 5.5 검증이 논문에 요구하는 서술 변경

1. **기관 momentum은 잡음**(부호 71%, CI 0 포함, 제거 시 유의하게 개선) →
   "기관은 momentum에 반응하지 않는다"로 서술. 계수 값을 해석하지 않는다.
2. **개인 underwater는 예측을 해치지만 계수는 매우 안정**(+0.0289, 100%, CI 0 배제) →
   구조 해석용 유지 방침(`experiments/2026-07-12/1840_최종모델_구조해석_5특징.md`)과
   일치. 이 트레이드오프를 본문에 명시한다.
3. **B는 유의하지 않다** → α(주효과) 중심 서술, B는 패턴만.
4. **2023년 개인 붕괴**를 한계 절에 기재한다.

---

## 6. 문헌 근거

**1차: Sias, R. W. (2002) *Institutional Herding*** (이후 *RFS* 17(1), 2004, 165–206)

Sias는 투자자 수요의 시차 상관을 두 성분으로 분해한다.

$$\beta\ (\text{전체}) = \underbrace{\text{자기 과거 거래를 따르는 성분}}_{\text{following own lag trades}} + \underbrace{\text{서로를 따르는 성분}}_{\text{true herding}}$$

실측(원자료 기준): 전체 **0.1193** = 자기추종 **0.0614** + herding **0.0579**.
즉 herding 측정치의 **약 절반이 자기 지속성**이며, 분해하지 않으면 오독하게 된다.
persist는 이 분해의 **자기추종 성분**에 해당한다.

### 서술 초안

> "Sias(2002)는 투자자 수요의 시차 상관이 자기추종과 상호추종으로 분해됨을 보이고,
> 전자가 절반가량을 차지함을 보고한다. 유형 수준 집계 데이터에서는 세 유형이 시장을
> 소진하여 $\sum_j u_j \approx 0$이 두 성분을 기계적으로 결합시키므로(상관 −0.99)
> 분해가 불가능하다. 따라서 우리는 측정 가능한 자기추종 성분만을 특징으로 사용하고,
> 유형 내 herding은 주장하지 않는다."

### 명명 원칙

**"herding"으로 부르지 않는다.** `flow persistence`(거래 지속성)로 표기한다.
합산 flow의 지속성은 "같은 주체가 계속 사는 것"과 "다른 주체들이 이어받는 것"을
구분하지 못하므로(Sias가 분해로 갈라낸 두 성분이 뭉쳐 있음), herding을 주장할 수 없다.

### 보조 근거 (서지 확인 필요)

- **주문 분할 집행(order splitting)** — 대량 주문이 여러 날에 걸쳐 실행되는 현상이
  flow 지속성의 직접적 원인. Kyle(1985) 전략적 거래자 모형, 최적 실행 이론(Almgren–Chriss),
  기관 주문 실행 실증(Chan–Lakonishok) 계열. ⚠ 서지·주장 미검증, 인용 전 원문 확인 필요.
- **Grinblatt–Keloharju(2001)** — 이미 검증된 문헌. 일별 핀란드 데이터로 유형별 매매를
  다루므로 지속성 관련 서술이 있을 가능성. PDF 보유 중이므로 확인 용이.

---

## 7. 남은 작업

1. ~~torch 정식 실행~~ → **완료**, §4 수치 확정 (2026-07-28)
2. ~~검증 스위트~~ → **완료**, §5.4 (2026-07-28)
3. **논문 HOLD 배너 제거** — `acmart-primary/mac_irl_icaif26.tex` L292–297 (미실행)
4. **§3·§4 작성** — 특징 정의·결과 보고 (미작성)
5. **§2 block 3 재작성** — herd 문단을 persist로 교체, Sias 인용 문맥 반영 (미실행)
6. 예상 부호 표 갱신 — persist는 **양(+)** 기대(AR(1) 양수), herd 시절 "방향 기대 없음"에서 변경
7. §5.5의 4개 서술 변경 반영 (기관 momentum / 개인 underwater 트레이드오프 / B 비유의 / 2023 한계)
8. 보조 문헌 서지 검증 (주문 분할 계열)
9. 레거시 config `features_underwater.yaml`·`features_no_volatility.yaml`이 `persist`를
   참조하나 의미가 바뀌었음 — 처리 방침 **사용자 판단 필요**
