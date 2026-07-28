# herd 피처 식별 문제 — 작업 브리프

> 작성 2026-07-27. **이 파일 하나만 읽으면 작업을 시작할 수 있도록** 자기완결적으로 씀.
> 관련: `docs/paper_checklist.md` → **A-H1**, `experiments/2026-07-27/1932_herd_자기시차_공선성_진단.md`
> 마감: ICAIF 2026, **2026-08-02 AOE** (8페이지, 참고문헌 포함)

---

## 30초 요약

`herd` 피처가 **해당 투자자 자신의 전일 흐름을 부호 반전한 것과 거의 같습니다 (r ≈ −0.99).**
따라서 herd 계수는 "타 유형에 대한 반응"이 아니라 **자기 흐름 지속성이 회계 항등식을 통해
재포장된 값**입니다. 부호 일관성 100%도 행동의 증거가 아니라 항등식의 필연입니다.

**좋은 소식: 논문의 헤드라인은 이미 안전한 것으로 확인됐습니다.** 기존 검증 스위트에
`remove_herd` 절제 실험이 이미 들어 있고, herd를 빼도 모멘텀 대비 구조가 그대로입니다
(아래 §3). 추가 실행 없이 확인된 사실입니다.

**필요한 것:** 세 번째 피처를 어떻게 할지 결정 → 재실행 → 논문 문장 갱신.

---

## 1. 원인

`src/data/preprocess.py:53`

```python
u_{investor} = net_buy_{investor} / trading_value
```

분모 `trading_value`가 **유형별이 아니라 종목 전체 거래대금**입니다. 세 유형이 분모를
공유하므로 시장청산이 `u` 공간에서 거의 그대로 성립합니다.

- 세 `u`의 합: 평균 −0.0095, 평균 절대값 **0.0133**
- 비교: 평균 `|u_foreign|` = **0.1254** → 잔차(기타법인·국가 등)는 스케일의 약 10%

`src/features/herd.py:9`가 herd를 나머지 두 유형의 전일 `u` 평균으로 만들므로:

$$\text{herd}_i(t) = \tfrac{1}{2}\Big(\sum_j u_j - u_i\Big)(t-1) \;\approx\; -\tfrac{1}{2}\,u_i(t-1)$$

---

## 2. 증거

**자기시차와의 상관**

| 유형 | corr(herd(t), 자기 u(t−1)) |
| --- | ---: |
| 외국인 | **−0.989** |
| 기관 | **−0.974** |
| 개인 | **−0.992** |

**계수가 기계적임을 보여주는 대조** — 자기상관이 양이므로 herd 계수는 항등식만으로도
음수가 되어야 하고, 크기는 AR(1) 순서를 따라야 합니다. 실제로 정확히 그렇습니다.

| 유형 | 자기 u의 AR(1) | β_herd | 부호 일관성 |
| --- | ---: | ---: | ---: |
| 외국인 | +0.394 | −0.0382 | 100% |
| 개인 | +0.365 | −0.0319 | 100% |
| 기관 | +0.169 | −0.0047 | 100% |

세 유형 모두 음수, 크기 순서가 AR(1) 순서와 일치.

**동시점 유형 간 상관** — 외국인–기관은 거의 직교(−0.048)이나 개인이 양쪽 상대 역할을 흡수.

|  | 외국인 | 기관 | 개인 |
| --- | ---: | ---: | ---: |
| 외국인 | 1.000 | −0.048 | −0.818 |
| 기관 | −0.048 | 1.000 | −0.522 |
| 개인 | −0.818 | −0.522 | 1.000 |

---

## 3. ⭐ 헤드라인은 이미 안전합니다 (추가 실행 불필요)

`runs/continuous_reward3_lambda_validation/ablation/remove_herd/`가 **이미 존재합니다.**
baseline과 비교하면:

| 유형 | 피처 | baseline | herd 제거 | 변화 | 일관성(base→제거) |
| --- | --- | ---: | ---: | ---: | ---: |
| 외국인 | momentum | +0.0476 | **+0.0639** | +0.0163 | 100% → 100% |
| 개인 | momentum | −0.0301 | **−0.0325** | −0.0024 | 100% → 100% |
| 기관 | momentum | −0.0013 | −0.0012 | ±0.0000 | 62% → 58% |
| 개인 | underwater | +0.0287 | +0.0314 | +0.0027 | 100% → 100% |
| 외국인 | underwater | +0.0050 | +0.0017 | −0.0033 | 89% → 76% |
| 기관 | underwater | −0.0021 | −0.0022 | −0.0001 | 93% → 91% |

**해석**

- **외국인 +모멘텀 vs 개인 −모멘텀, 양쪽 100% 일관성 — 완전히 유지됩니다.** 논문의
  핵심 주장은 herd 명세와 무관합니다.
- 외국인 모멘텀이 오히려 **커집니다**(+0.0476 → +0.0639). herd가 모멘텀과 상관된
  자기 지속성을 흡수하고 있었다는 §1 진단과 정확히 일치합니다
  (corr(자기시차, mom20) = +0.425).
- 개인 underwater(처분효과)는 100% 유지, 외국인은 89% → 76%로 더 약해짐 →
  **"처분효과는 개인만"이라는 대조가 오히려 선명해집니다.**
- 기관은 어느 쪽이든 null.

즉 이 문제는 **논문을 위협하는 사안이 아니라 세 번째 피처의 해석 문제**입니다.

---

## 4. 확실한 것 / 불확실한 것

**확실 (구조적, 파이프라인 무관)**

- 공통 분모 → 세 `u`가 거의 상계 → herd ≈ −½·자기시차
- 자기상관 부호와 순서
- `remove_herd` 절제 결과 (실제 run 산출물)

**불확실**

- §2의 상관·AR(1) 수치는 raw CSV에서 `u`·`mom20`을 **재구성**하고 2022-01-01 컷오프를
  적용해 **n = 977**로 계산. canonical run은 **n = 973**
  (`sample: start 2020-01-01, end 2025-12-31, size 973, selection: latest`).
  → **논문에 넣기 전 canonical 피처 행렬로 재계산 필요.**

---

## 5. 선택지

| | 내용 | 비용 | 평가 |
| --- | --- | --- | --- |
| **A** | 이름만 "자기 흐름 지속성(부호 반전)"으로 재해석, 재실행 없음 | 없음 | 피처가 여전히 *타 유형* 흐름이라, 상관을 계산해 본 심사자에게 "왜 자기 시차를 안 썼나" 답변 불가. **비추천** |
| **B** | `herd` → `persist` 교체 | 낮음 (아래 참조) | 3피처 유지, 계수 전부 해석 가능, 청산 교란 제거. ⚠️ Oh(2025)는 지속성이 **개인** 최강이라 하나 우리 AR(1)은 **외국인** 최강 — DFA Hurst ≠ AR(1), 표본·종목 상이. §2 문장 수정 필요 |
| **C** | 타 유형 하나만 사용 (`herd_a`/`herd_b`, 구현됨) | 낮음 | 외국인–기관 쌍(−0.048)은 진짜 식별되나 개인 쌍은 −0.818/−0.522. **유형별 식별 품질이 달라져 "대칭적 추정" 셀링포인트 훼손.** 비추천 |
| **D** | 2피처로 축소 (herd 제거) | 없음 — **결과가 이미 있음** | §3 표가 곧 이 결과. 가장 정직하고 가장 쌈. 단 행동 근거 하나(herding)를 잃고, §2 herding 문헌 블록의 역할이 줄어듦 |

**추천: B 또는 D.** 둘 다 돌려 비교한 뒤 정하는 것이 가장 안전합니다. D는 이미 결과가
있으니 사실상 B만 실행하면 비교가 완성됩니다.

### ⚠️ B를 택할 경우 — `persist`는 아직 등록되어 있지 않습니다

`src/features/persist.py`에 `build_persist`는 구현돼 있으나
**`src/features/registry.py`의 `FEATURE_REGISTRY`에 없습니다.** 두 줄 추가 필요:

```python
from src.features.persist import build_persist   # import 블록에
...
    "persist": build_persist,                     # FEATURE_REGISTRY 안에
```

---

## 6. 실행 계획

### 단계 1 — 진단 수치 확정 (필수, 선행)

canonical 피처 행렬(`runs/continuous_reward3_lambda_unified/`, n=973)로 §2의 상관·AR(1)을
재계산. raw CSV 재구성이 아니라 실제 파이프라인 산출물 사용.

### 단계 2 — B 설정 준비

1. 위 registry 두 줄 추가
2. `configs/features_continuous_reward3.yaml`를 복사해
   `configs/features_continuous_persist3.yaml` 생성, `herd` → `persist`로 교체
3. 실험 config의 `experiment.name` / `output_dir`을
   `continuous_persist3_lambda_unified`로 변경

### 단계 3 — 재실행

기준 설정은 `runs/continuous_reward3_lambda_unified/config_snapshot.yaml`에 전부 있습니다
(재실행 시 이 스냅샷을 출발점으로 쓰는 것이 가장 안전).

```bash
python scripts/train.py \
  --data-config configs/data_continuous.yaml \
  --features-config configs/features_continuous_persist3.yaml \
  --model-config configs/model.yaml \
  --train-config configs/train.yaml \
  --experiment-config <새 experiment yaml>
```

검증 스위트:

```bash
python scripts/run_continuous_reward_validation.py \
  --features-config configs/features_continuous_persist3.yaml \
  --output-root runs/continuous_persist3_validation \
  --bootstrap-resamples 200
```

⚠️ 배시 호출이 45초에서 끊깁니다. 검증 스크립트는 resume 로직이 있어 반복 호출하면
이어서 진행되지만, **부트스트랩은 resume이 없어 한 번의 호출 안에서 끝나야 합니다.**

### 단계 4 — 비교 (헤드라인 보호용 핵심 점검)

세 설정의 momentum·underwater 계수와 부호 일관성을 나란히 비교:
baseline(herd) / remove_herd(=D) / persist(=B).

### 단계 5 — 문서 갱신

- `experiments/2026-07-27/` 또는 실행일자 폴더에 한국어 실험 노트 (CLAUDE.md 규칙 5)
- `docs/paper_checklist.md` A-H1 갱신
- 아래 §7의 동결 문장들 해제 및 재작성

---

## 7. 논문 쪽 파급 — 이 결정이 풀어야 하는 동결 항목

| 위치 | 상태 |
| --- | --- |
| §2 "Behavioral regularities behind type differences" 블록 | herding 문헌을 소개 중. **D를 택하면 herding 인용의 명분이 약해짐** |
| §2 블록 ①의 Oh(2025) 문장 | "persistence strongest for **retail**, weakest for **foreign**" — **B를 택하면 우리 AR(1)(외국인 최강)과 정면 충돌.** 순서를 빼는 수정안 대기 중 |
| §3 이동 문단의 herd 문장 | 🔴 동결. 시장청산을 기제로 대는데 **틀린 기제** |
| §3 이동 문단의 마지막 문장 | 🔴 동결. "momentum과 loss region에만 예상 부호" 대조가 **B에서는 무너짐**(지속성은 **+** 예상) |
| §3.4 Table 1 세 번째 행 | 🔴 미작성 |
| 초록 / 기여 ③ | ✅ 2026-07-27에 `beyond herding` 삭제 완료 — 어느 옵션에도 안전 |

---

## 8. 작업 중 발견한 별건 (같이 처리 권장)

1. **부트스트랩 100회.** `runs/.../weight_bootstrap/bootstrap_reward_weights_summary.csv`의
   `n_resamples = 100`. 스크립트 기본값은 200이므로 명시적으로 100을 준 것. 제출 전 200으로.
2. **`validation_manifest.yaml`의 라벨이 낡음** — "continuous **5-feature** contextual model"
   이라 적혀 있으나 실제 피처는 3개(momentum, herd, underwater).
3. **하이퍼파라미터가 유형별로 다릅니다.** λ는 0.005로 통일됐지만
   epochs 75/10/20, lr 0.001/0.0005/0.001, batch 256/2048/512 (foreign/institution/retail).
   논문은 "identical conditions"라 서술 → **심사자가 "기관 null은 10 epoch 미학습 아니냐"고
   물을 수 있음.** 초록의 "despite converged optimization"을 뒷받침할 수렴 근거가 §3.5에
   필요합니다. A-H1과 별개지만 성격이 같은(기관 null의 진위) 취약점.
4. **`contexts.selected`에 `vkospi_1d`가 있으나 `model.context_names`에는 없음** — 생성되지만
   미사용. 의도된 것인지 확인 필요.
5. **누락된 파일들.** `experiments/2026-07-24/`, `experiments/2026-07-26/`가 디스크에 없습니다
   (`experiments/`가 07-19에서 07-27로 건너뜀). `docs/`에도 `paper_ch1_en.md`,
   `paper_ch1_kr.md`, `refs_checklist.md`가 없습니다. `docs/`는 한 번도 커밋된 적이 없어
   git 복구 불가. **원고 `.tex`는 온전하므로 논문 내용 손실은 없습니다.**
   `paper_checklist.md`의 일부 경로 참조가 깨진 링크입니다.

---

## 9. 재현 코드 (진단)

```python
import pandas as pd, numpy as np
df = pd.read_csv('data/raw/samsung_macirl_EXTENDED_2019_2025.csv')
df['date'] = pd.to_datetime(df['date']); df = df.sort_values('date')
tv = df['trading_value'].replace(0, np.nan)
inv = ['foreign', 'institution', 'retail']
for i in inv:
    df['u_' + i] = (df[i + '_buy_value'] - df[i + '_sell_value']) / tv
df['mom20'] = df['close'].pct_change(20)
d = df.dropna(subset=['u_foreign', 'u_institution', 'u_retail', 'mom20'])
d = d[d['date'] >= '2022-01-01'].reset_index(drop=True)

print((d['u_foreign'] + d['u_institution'] + d['u_retail']).abs().mean())
print(d[['u_foreign', 'u_institution', 'u_retail']].corr())
for i in inv:
    o = [j for j in inv if j != i]
    herd = (d['u_' + o[0]].shift(1) + d['u_' + o[1]].shift(1)) / 2
    print(i, herd.corr(d['u_' + i].shift(1)), d['u_' + i].autocorr(1),
          herd.corr(d['mom20'].shift(1)))
```

절제 비교:

```python
import csv
def load(p):
    return {(r['investor'], r['feature']): (float(r['mean']), float(r['direction_consistency']))
            for r in csv.DictReader(open(p))}
b = load('runs/continuous_reward3_lambda_validation/ablation/baseline/reward_weights_summary.csv')
h = load('runs/continuous_reward3_lambda_validation/ablation/remove_herd/reward_weights_summary.csv')
for k in sorted(set(b) & set(h)):
    print(k, b[k], h[k])
```

---

## 10. 관련 파일

| 용도 | 경로 |
| --- | --- |
| 피처 정의 | `src/features/herd.py`, `src/features/persist.py` |
| 등록부 | `src/features/registry.py` ← **`persist` 미등록** |
| 전처리 (`u` 정의) | `src/data/preprocess.py:48-53` |
| 현재 설정 | `configs/features_continuous_reward3.yaml` |
| canonical run | `runs/continuous_reward3_lambda_unified/` (설정 전량: `config_snapshot.yaml`) |
| 검증 스위트 | `runs/continuous_reward3_lambda_validation/` |
| **절제 결과 (D안)** | `runs/continuous_reward3_lambda_validation/ablation/remove_herd/` |
| 학습 스크립트 | `scripts/train.py` |
| 검증 스크립트 | `scripts/run_continuous_reward_validation.py` |
| 실험 노트 | `experiments/2026-07-27/1932_herd_자기시차_공선성_진단.md` |
| 체크리스트 | `docs/paper_checklist.md` → A-H1 |
| 원고 | `../acmart-primary/mac_irl_icaif26.tex` |
