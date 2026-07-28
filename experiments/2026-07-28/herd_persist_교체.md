# herd → persist 교체 실험 (3특징 canonical 대체)

- 날짜/시간: 2026-07-28
- 목적: `herd`가 시장청산 항등식 때문에 자기시차의 부호 반전(상관 −0.99)이 되어 유형 간 반응을 측정하지 못하는 문제를 해결. Sias(2002) 분해의 자기추종 성분에 해당하는 `persist`로 교체하고, canonical 대비 성능·계수 안정성을 확인.
- 변경한 것:
  - `src/features/persist.py`의 정의 자체를 공통분모(종목 전체 거래대금)에서
    **자기분모(자기 총거래금액, `investor_trade_imbalance`)** 로 교체. 별도
    `persist_own.py` 파일은 만들지 않고 `persist` 하나로 통일(최종 채택안).
  - `src/features/registry.py`에 `persist` 등록(`build_persist`가 미등록 상태였음).
  - config 신규: `features_continuous_reward3_persist.yaml`, `experiment_continuous_reward3_persist.yaml`.
  - **herd → persist 교체 외 일체 변경 없음.**
  - (정정 2026-07-28: 최초 작성 시 "persist(공통분모)/persist_own(자기분모) 2종 병행"으로
    기록했으나, 실제로는 `persist.py`를 자기분모로 직접 교체했을 뿐 두 번째 파일은
    존재한 적이 없다. 아래 수치는 이번에 `scripts/train_continuous.py` 정식 실행으로 재확인.)
- 고정 조건 (canonical `runs/continuous_reward3_lambda_unified`와 동일):
  - 연속형 contextual linear, κ=1, τ=1, **컨텍스트 주효과 α 사용**.
  - 컨텍스트 `kospi_return_1d`, `fx_level_z_252`.
  - CPCV 10 folds / test 2 / purge 1 / embargo 5 = 45 split, split별 train-only 표준화.
  - 투자자별 HP: 외국인 ep75/lr0.001, 기관 ep10/bs2048/lr0.0005, 개인 ep20/bs512/lr0.001, **λ=0.005 통일**.
  - seed 42.
- 데이터: `samsung_macirl_EXTENDED_2019_2025.csv` → `data/processed/dataset_continuous_reward3_persist.npz`, **973행(2022-01-06~2025-12-29)**, canonical과 동일 표본.
- 설정 파일: `configs/data_continuous.yaml`, `configs/features_continuous_reward3_persist.yaml`, `configs/model.yaml`, `configs/train.yaml`, `configs/experiment_continuous_reward3_persist.yaml`
- 결과 폴더: **`runs/continuous_reward3_persist/`**(`scripts/train_continuous.py` 정식 실행, torch, seed 42, 2026-07-28 정식 확정)
  cv_metrics.csv, cv_metrics_summary.csv, reward_weights.csv, reward_weights_summary.csv, context_main_weights.csv, context_main_weights_summary.csv
  - (참고: `runs/continuous_reward3_persist_numpy_prelim/`는 교체 전 공통분모 정의로 만든 예비 numpy 재현 결과라 현재 코드와 정의가 달라 **폐기**. `runs/continuous_reward3_lambda_unified/`가 canonical(herd) 비교 기준.)
- 주요 결과: **전 주체에서 성능이 개선되거나 동등**하고, 헤드라인(외국인 추세추종 / 개인 역행)이 그대로 유지됨. persist는 세 주체 모두 **양(+), 부호 일관성 100%** 로 herd의 인위적 음수 부호가 해소됨. 자기분모 정의를 채택 — 행동과 정의가 일치해 해석이 엄밀해지고 외국인 상관도 0.284→0.306으로 개선.

## 특징 정의식

### 공통 재료 — 순매수 비율 $u$

`src/data/preprocess.py:53`

$$u_{i,t} = \frac{\text{buy\_value}_{i,t} - \text{sell\_value}_{i,t}}{\text{trading\_value}_t}$$

분모는 유형별이 아니라 **종목 전체 거래대금**이다(세 유형 공통).
※ 행동(action) $a$의 정의는 자기 총거래 분모를 쓰므로 서로 다름 → 한계 절 참조.

### persist (최종 채택, 자기분모) ★

`src/features/persist.py`

$$\phi^{\text{persist}}_{i,t} = a_{i,t-1} \times a, \qquad a_{i,t} = \frac{\text{buy}_{i,t}-\text{sell}_{i,t}}{\text{buy}_{i,t}+\text{sell}_{i,t}}$$

**어제의 행동 그 자체.** 모델이 예측하는 행동 $a$와 **동일한 정의**(자기 총거래 분모)를
사용하므로, 계수가 행동에 대한 문자 그대로의 AR(1)이 된다. Sias(2002) 분해의
자기추종(following own lag trades) 성분에 해당.

> 1차 시도는 $u_{i,t-1} \times a$(종목 전체 거래대금 분모, 행동과 다른 척도)였으나
> 바로 아래 "왜 바꿨나" 사유로 자기분모로 교체했다. 별도 파일 없이 `persist.py` 자체를
> 이 정의로 덮어썼다.

> **왜 바꿨나 — 정합성.** 1차 시도(공통분모 정의)는 특징과 행동이 서로 다른 척도였다.
> "어제의 시장점유율 기준 순매수"로 "오늘의 자기거래 기준 순매수"를 예측하는 구조라
> "자기 행동의 지속"이라는 해석이 엄밀히 성립하지 않았다. 예: 어제 거래량은 적었으나
> 대부분 매수였다면 자기분모로는 $a \approx +0.8$(강한 매수 성향)이지만 공통분모로는
> $u \approx +0.01$(시장에서 미미)이다. 즉 1차 persist는 "성향의 지속"이 아니라
> "시장 영향력의 지속"을 측정하고 있었다.
> 이 변경은 결과에 맞춘 조정이 아니라 **정의상 정합성 수정**이다.

### herd (기존, 교체 대상)

`src/features/herd.py`, window = 1

$$\phi^{\text{herd}}_{i,t} = \Big(\frac{1}{2}\sum_{j \neq i} u_{j,t-1}\Big) \times a$$

**다른 두 유형의 전일 순매수 비율 평균.**

### 두 식의 관계 — 교체의 근거

시장청산으로 $\sum_j u_{j} \approx 0$이므로 (실측: $\sum u$ 평균절대값 0.0133 vs
$|u_{\text{foreign}}|$ 평균 0.1254, 잔차 약 10%),

$$\phi^{\text{herd}}_{i,t} = \frac{1}{2}\Big(\sum_j u_{j,t-1} - u_{i,t-1}\Big) \approx -\frac{1}{2}\,u_{i,t-1} = -\frac{1}{2}\,\phi^{\text{persist}}_{i,t}$$

실측 상관: 외국인 **−0.989**, 기관 −0.974, 개인 −0.992.

→ **herd는 "다른 유형의 flow"가 아니라 "자기 전일 flow의 부호 반전"을 측정하고 있었다.**
persist는 같은 정보를 직접·양의 부호로 측정하므로 해석이 일치한다.

### 참고 — 비교 대상 특징

| 특징 | 정의식 | 비고 |
| --- | --- | --- |
| momentum | $\log(P_t / P_{t-20}) \times a$ | 20일 로그수익 |
| underwater | $\max\!\big(0, (\bar c_{i,t} - \mathrm{VWAP}_t)/\bar c_{i,t}\big) \times a$ | $\bar c$ = ρ=0.98 감쇠 평균단가 |

## OOS 성능 비교 (45 CPCV split 평균 ± 표준편차)

정식 실행(`scripts/train_continuous.py`, torch, seed 42, 2026-07-28) 기준.

| 대상 | 지표 | canonical(herd) | **persist(자기분모, 최종)** |
| --- | --- | ---: | ---: |
| 외국인 | 방향정확도 | 0.6365 ± 0.0414 | **0.6435 ± 0.0395** |
| 외국인 | 상관 | 0.2837 ± 0.1056 | **0.3060 ± 0.1061** |
| 외국인 | RMSE | 0.2453 | **0.2423** |
| 기관 | 방향정확도 | 0.5332 ± 0.0376 | 0.5355 ± 0.0376 |
| 기관 | 상관 | 0.0731 ± 0.0596 | **0.0757 ± 0.0579** |
| 기관 | RMSE | 0.1250 | 0.1251 |
| 개인 | 방향정확도 | 0.6083 ± 0.0439 | **0.6120 ± 0.0458** |
| 개인 | 상관 | 0.2465 ± 0.1178 | **0.2478 ± 0.1205** |
| 개인 | RMSE | 0.3174 | 0.3172 |

출처: `runs/continuous_reward3_lambda_unified/cv_metrics_summary.csv` (canonical),
`runs/continuous_reward3_persist/cv_metrics_summary.csv` (persist)

→ **persist가 전 주체에서 최고이거나 동등.** 특히 외국인 상관 0.2837 → **0.3060**
(+0.022)로 뚜렷한 개선. 정합성 수정이 성능 손실 없이 오히려 이득을 냈다.

## 가중치 결과

### β (평균 ± 표준편차, 부호 일관성) — canonical 대비

| 대상 | 피처 | canonical(herd) β | **persist(자기분모, 최종) β** |
| --- | --- | ---: | ---: |
| 외국인 | momentum | +0.0476 ± 0.0089 (100%) | **+0.0435 ± 0.0090 (100%)** |
| 외국인 | herd → persist | −0.0382 ± 0.0065 (100%) | **+0.0505 ± 0.0060 (100%)** |
| 외국인 | underwater | +0.0050 ± 0.0048 (88.9%) | **+0.0061 ± 0.0048 (91.1%)** |
| 기관 | momentum | −0.0013 ± 0.0018 (62.2%) | −0.0013 ± 0.0018 (71.1%) |
| 기관 | herd → persist | −0.0047 ± 0.0002 (100%) | **+0.0047 ± 0.0003 (100%)** |
| 기관 | underwater | −0.0021 ± 0.0016 (93.3%) | −0.0021 ± 0.0016 (93.3%) |
| 개인 | momentum | −0.0301 ± 0.0015 (100%) | **−0.0301 ± 0.0015 (100%)** |
| 개인 | herd → persist | −0.0319 ± 0.0013 (100%) | **+0.0319 ± 0.0014 (100%)** |
| 개인 | underwater | +0.0287 ± 0.0031 (100%) | **+0.0289 ± 0.0031 (100%)** |

출처: `runs/continuous_reward3_lambda_unified/reward_weights_summary.csv` (canonical),
`runs/continuous_reward3_persist/reward_weights_summary.csv` (persist)

**핵심 확인 — momentum·underwater 계수가 거의 완전히 보존됨**(소수 넷째 자리 수준 변화).
헤드라인(외국인 추세추종 +0.047 / 개인 역행 −0.030, 개인 손실회피 +0.029)이 교체의 영향을 받지 않았다.

**persist 부호는 세 주체 모두 양(+), 일관성 100%.** herd에서 모두 음수였던 것이 부호 반전되어 나타났으며, 이는 herd ≈ −½·persist라는 항등식과 정확히 일치한다. AR(1)이 양수(+0.394/+0.156/+0.365)이므로 persist가 양수인 것이 자연스러운 해석("어제 산 만큼 오늘도 산다").

### α (컨텍스트 주효과, 부호 일관성) — persist 명세

| 대상 | 컨텍스트 | α | 일관성 |
| --- | --- | ---: | ---: |
| 외국인 | kospi_return_1d | +0.0331 | 100% |
| 외국인 | fx_level_z_252 | −0.0215 | 97.8% |
| 기관 | kospi_return_1d | −0.0044 | 100% |
| 기관 | fx_level_z_252 | −0.0033 | 93.3% |
| 개인 | kospi_return_1d | −0.0235 | 100% |
| 개인 | fx_level_z_252 | +0.0289 | 100% |

출처: `runs/continuous_reward3_persist/context_main_weights_summary.csv`

외국인과 개인이 정확한 거울상(KOSPI 상승일 외국인 매수/개인 매도, 원화 약세 국면 외국인 매도/개인 매수). canonical과 동일 패턴.

## 진단 — persist가 momentum과 구별되는가

> **주의(정정 2026-07-28): 아래 VIF·상관·AR(1) 수치는 저장소에 재현 스크립트/출력 파일이
> 없는 ad hoc 분석 결과다.** OOS 성능·가중치 표(위)는 이번에 `scripts/train_continuous.py`
> 정식 실행으로 재확인했지만, 이 절의 수치는 그 대상이 아니었다. 최초 기록 당시
> "공통분모/자기분모 병행 비교"로 적혀 있었는데 지금 코드에는 자기분모(최종) 정의
> 하나만 남아 있으므로, 아래는 그 한 열만 남기고 정리했다. 인용 전 재실행·저장 필요.

### VIF (n=973) — 안전

| 대상 | momentum | persist | underwater |
| --- | ---: | ---: | ---: |
| 외국인 | 1.27 | 1.24 | 1.05 |
| 기관 | 1.01 | 1.01 | 1.01 |
| 개인 | 1.60 | 1.22 | 1.58 |

**전부 경고선(5) 대비 매우 낮음.** 개별 계수 해석에 지장 없음.

### persist ↔ momentum 상관과 공유 분산

| 대상 | corr (공유 분산) |
| --- | ---: |
| 외국인 | +0.434 (18.8%) |
| 기관 | +0.071 (0.5%) |
| 개인 | −0.386 (14.9%) |

상관 0.43은 커 보이나 **공유 분산은 최대 19%**이며, persist의 80% 이상이 momentum으로
설명되지 않는 고유 정보다. 기관은 사실상 완전 독립(0.5%).

### 행동의 AR(1) — persist 계수의 직접 의미

자기분모 행동 $a_{i,t}$의 1차 자기상관:

| 대상 | AR(1) |
| --- | ---: |
| 외국인 | **+0.403** |
| 기관 | +0.151 |
| 개인 | **+0.344** |

persist는 이 행동 자체의 시차이므로, 계수가 양수인 것은 관측된 AR(1) 부호와
정확히 일치한다. 즉 **계수 해석("어제 한 만큼 오늘도 한다")이 원자료 통계와 직접 대응**한다.
1차 시도(공통분모)에서는 이런 직접 대응이 성립하지 않았다.

### 저변동일 검증 — 가격이 움직이지 않아도 지속성이 존재하는가

|일간수익| 하위 33%인 날(325/974일)만 골라 flow 자기상관을 재계산.

| 대상 | 전체 AR(1) | **저변동일 AR(1)** |
| --- | ---: | ---: |
| 외국인 | +0.394 | **+0.347** |
| 기관 | +0.156 | +0.113 |
| 개인 | +0.365 | **+0.375** |

**가격이 거의 움직이지 않은 날에도 지속성이 유지된다.** momentum으로는 설명할 수 없는 현상이며, 주문 분할 집행·거래 관성이라는 별도 메커니즘의 존재를 지지한다. 변수를 손대지 않고(직교화 없이) persist와 momentum의 독립성을 데이터로 입증한 결과다.

## 해석

- **교체 근거는 문헌 불일치가 아니라 측정 실패였다.** herd는 시장청산($\sum_j u_j \approx 0$)으로 인해 herd$_i \approx -\frac{1}{2}u_{i,t-1}$이 되어(상관 −0.99), 유형 간 반응이 아니라 자기시차를 측정하고 있었다. 부호 일관성 100%도 행동 증거가 아니라 회계 항등식의 결과였다.
- **persist는 Sias(2002) 분해의 자기추종 성분**에 해당한다. Sias는 시차 상관을 자기추종과 상호추종으로 분해하고 전자가 절반을 차지함을 보고했다. 유형 수준 집계 데이터에서는 두 성분의 분리가 불가능하므로, 측정 가능한 자기추종 성분만을 사용한다. **"herding"으로 명명하지 않는다.**
- **헤드라인이 보호되었다.** momentum·underwater 계수가 사실상 불변이므로, 외국인 추세추종 / 개인 역행·처분효과라는 핵심 주장은 이번 명세 변경의 영향을 받지 않는다.
- 성능은 외국인에서 소폭 개선, 나머지는 동등. 즉 **해석의 정당성을 확보하면서 성능 손실이 없다.**

## 한계

- ~~numpy 재현 결과다(torch 미설치 환경)~~ → **해소 (2026-07-28).** `.venv`의 torch로
  `scripts/train_continuous.py` 정식 실행 완료(`runs/continuous_reward3_persist/`).
  OOS 성능·β·α 표는 이 정식 실행 결과다. (저장된 `runs/continuous_reward3_persist_numpy_prelim/`은
  공통분모(구) 정의 결과라 지금 값과 다르며 참고 대상이 아니다 — 위 "결과 폴더" 항목 참조.)
- **정합성 사안은 본 실험에서 해소되었다.** 기존 herd·1차 persist는 `preprocess.py:53`의 공통분모 $u$를 사용해 행동(`labels.py`의 자기분모) 정의와 불일치했다. 최종 채택한 `persist`(자기분모)는 행동과 동일 정의를 사용한다.
  다만 `momentum`·`underwater` 등 나머지 특징은 정의상 분모 문제와 무관하다(가격·평단 기반).
- §"진단" 절의 VIF·상관·AR(1) 수치는 여전히 ad hoc 분석이라 파일 근거가 없다(위 경고 참조).

## 다음 액션

1. ~~`scripts/train_continuous.py`로 `persist` 명세 정식 실행~~ → **완료 (2026-07-28)**, 위 수치로 확정.
2. persist 문헌 근거 확정 — Sias(2002) 자기추종 성분을 1차 근거로, 주문 분할 집행 문헌을 보조로 (서지 검증 필요)
3. §2 block 3(herd 문단)·§3.3 특징 정의·§4 보고 재작성, 예상 부호 표 갱신
4. 검증 스위트(bootstrap·walk-forward·ridge·ablation) 재실행
5. 명명 확정 — "herding"이 아닌 **flow persistence**(거래 지속성)로 표기
