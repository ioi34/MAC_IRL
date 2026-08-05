# §3 Method 재작성 — 논의 기록 (2026-07-31)

> **성격: 집필 결정 노트. 실험 아님.** 새로 실행한 run이 없으므로 `CLAUDE.md` §5의
> 가중치 표는 해당 없음. 아래 수치는 모두 기존 산출물(`runs/`,
> `experiments/2026-07-31/exact_lasso/`)과 코드 대조에서 인용한 것이다.
>
> 원고: `acmart-primary/mac_irl_icaif26.tex` (= `paper/mac_irl_icaif26.tex`)
> 선행: `docs/paper_ch3_plan.md` (v2, 2026-07-28) — 본 노트는 그 v2의 부분 개정
> 대조 문서: `SPR_4page_research_paper_writing_skill_20260728.pdf` (유민 초안)
> 근거 실행: `runs/continuous_reward3_persist_epochs75/`,
> `experiments/2026-07-31/exact_lasso/`
> 상태: **원고 미수정.** 결정만 확정, 집필 보류.

---

## 1. 재작성 사유

SPR 초안과 대조한 결과 §3에 세 가지 문제가 확인되었다.

1. **순서.** SPR은 1.1 문제 → 1.2 피처 → 1.3 모델 순인데, 원고는 3.1 setup →
   3.2 보상 → 3.3 모델 → **3.4 피처**다. 식 (2)~(5)에서 $x_{i,t}$, $C_t$가
   미정의 기호로 2쪽간 사용된다.
2. **분량.** §3 본문 ≈2,650단어(p.2~p.5 중반 ≈2.5쪽). SPR §1은 ≈1,030단어.
   PDF는 이미 8쪽 한도에 도달.
3. **성격.** 늘어난 부분이 설명이 아니라 **정당화 산문**이다. 논리적으로 보강하고
   쪼개다가 생긴 결과.

### 절별 분량 (단어, 그림 제외)

| 절 | 단어 | SPR 대응 |
| --- | ---: | --- |
| 3.1 Setup | 323 | 1.1 |
| 3.2 Reward and optimal action | 279 | 없음 |
| 3.3 Parameterization and reduction | 458 | 1.3 |
| 3.4 Features and sign expectations | 601 | **1.2** |
| 3.5 Estimation and uniqueness | 374 | 1.4 |
| 3.6 Validation protocol | 615 | 1.4 |
| **합계** | **2,650** | **1,030** |

---

## 2. 확정 결정

### D6. 구조 — SPR 순서를 따른다

```
3   Method (도입 4~5문장: 로드맵 + "conditional association, not structural preference")
3.1 Problem setup                       ~200   (데이터 문단 삭제)
3.2 State features and market context   ~500   ← 현행 3.4를 앞으로
3.3 Reward and policy                   ~350   ← 현행 3.2+3.3 병합
3.4 Estimation and validation           ~600   ← 현행 3.5+3.6 병합
```

목표 ≈1,650단어(−38%, 약 0.7~0.9쪽 확보).
3.5/3.6 병합은 SPR 1.4가 추정+V1–V4를 한 절로 두는 것과 동일.

### D7. 수식 취사 기준 — "코드가 실행했는가"가 아니라 **"무언가를 구속하는가"**

논의 중 "코드에서의 유도를 거쳐야만 넣을 수 있는가"라는 반문이 있었고, **아니다**로
결론. 해석적으로 푼 해를 구현하는 것은 정상이며 `continuous.py:40` docstring도
`"Closed-form continuous-action policy with kappa fixed to 1"`이다. 기준은 그
수식이 다른 무언가를 결정하느냐다.

| 수식 | 판정 | 근거 |
| --- | --- | --- |
| $R_i=g_{i,t}a-\tfrac12\kappa a^2$ | **유지** | clip±1과 선형 형태를 결정. 비용 형태를 바꾸면 구현이 바뀐다 |
| $a^*_{i,t+1}=\arg\max_{a\in\mathcal{A}}R_i$ | **추가** | 현재 미정의. 이것이 있어야 clip이 유도된다 |
| $\mathcal{A}=[-1,1]$ | **추가** | 현재 미선언. clip의 출처 |
| $\kappa$, $\kappa=1$ | **유지** | 가정 아닌 정규화. 유형 간 크기 비교의 근거 |
| 가우시안 정책 $\pi_i$, $\sigma_i$ | **삭제** | 구속하는 것 없음. MSE 사후 정당화. 저장소에 0회 등장 |

### D8. $\lambda=0.005$ 공통 유지

현행 3.5 서술 유지. SPR의 유형별 λ(0, 0.01, 0.0003)는 채택하지 않는다.
공통 λ가 계수 크기 비교를 가능하게 한다는 논거도 그대로.

### D9. 집필 방식 — 감산이 아니라 **가산**

현행 §3을 깎는 대신 **SPR 골격을 이식하고 델타만 더한다.**

- 감산은 원본의 잘못된 순서를 보존한 채 흉터를 남긴다. 가산은 올바른 순서에서
  시작하므로 D6의 구조 문제가 구조적으로 해결된다.
- **문안 재사용 제약 없음**: SPR은 같은 논문의 공저자 내부 초안 (2026-07-31 확인).
  문장 단위 직접 이식 가능.

**단, "SPR 그대로"는 아니다.** SPR에도 이미 진단한 결함이 있어 이식 시 제외한다.

| 제외 대상 | 사유 |
| --- | --- |
| SPR 1.4 *"The objective in Eq. (4) is convex, and coordinate descent obtains a global solution."* | §4.3의 3중 비약이 SPR에도 동일하게 존재 |
| SPR 1.4 $\lambda=0$ 분기 + KKT 벡터·등상관집합 $E_i$ | §5. 완전계수 rank 11이 모든 λ를 덮음 → SPR보다 **더** 줄이는 자리 |

#### 델타 표

| SPR § | 그대로 | 교체 | 추가 |
| --- | --- | --- | --- |
| 도입 | 로드맵 형식 | "Symmetric" → **"Shared-specification"** (D2) | — |
| 1.1 | 식 (1) $u_{i,t}$, 목표 $a_{t+1}$, 상태 $s=(x,C)$, 대칭성 | 절 제목 "Behavior Forecasting" — 예측 프레임이라 IRL과 충돌, 재명명 | $\mathcal{A}=[-1,1]$ 선언 |
| 1.2 | 모멘텀, underwater 재귀식 전부, 맥락 $r_t$·$z^{FX}$, 표준화 | **$x^{cross}$(타 유형 시차) → $x^{per}=a_{i,t-1}$(자기 시차)**. 근거도 Lakonishok herding → Sias 2004 own-following + Oh 2025 | 유형별 **사전 부호 기대** (SPR에 없음) |
| 1.3 | 식 (3), $w$·$\theta$ 벡터화, $\theta\in\mathbb{R}^{11}$, $\hat a=\mathrm{clip}$, "β·α만 해석" | $q_{i,t}$(linear score) → $g_{i,t}$(보상 기울기) | **보상 블록**: $R_i$, $\arg\max_{a\in\mathcal{A}}$, 사영 유도, $\kappa=1$ (§7 문안) |
| 1.4 추정 | 식 (4) Lasso, $\mathcal{D}_i$, rank 11·조건수 | 유형별 λ → **공통 0.005** + 비교가능성 논거 | **포화율 0** → (5)↔코드 일치 1문장 |
| 1.4 검증 | V1 CPCV(10/2, purge 1, embargo 5, 45 split), V2 월블록, V4 벌점 대체 | V2 1,000회 → **200회**. V3 10창(400 시작·60일 전진) → **3창(2023/24/25 역년)** | **피처 ablation + FDR**, **identified/unidentified 사전 기준**(V1 부호 ≥90% ∧ V2 구간 0 배제) |
| 1.4 지표 | 방향정확도·상관·MAE·OOS $R^2$, MAE 유형 순위 금지 | — | RMSE |
| 데이터 | **SPR §2.1에 있음 → §3에 안 들어옴** | — | — |

마지막 행이 공짜 이득 — 현행 3.1의 삼성전자·973일 문단이 자동 소거된다.

#### 예상 분량

| | 단어 |
| --- | ---: |
| SPR §1 기준 | 1,030 |
| + 보상/argmax/κ 블록 | +150 |
| + 사전 부호 기대 | +80 |
| + ablation·FDR | +90 |
| + identified 기준 | +70 |
| + 포화율 0 | +30 |
| − 볼록성 문장·등상관집합 | −70 |
| − V3 축소(10창→3창) | −20 |
| **합계** | **≈1,360** |

> **[2026-07-31 실측으로 정정]** 위 추정은 **틀렸다.** 실제 초안
> `docs/paper_ch3_draft_v3.tex`는 **1,995단어**(현행 2,486 대비 **−20%**).
>
> 추정이 빗나간 이유: "SPR 1,030 + 델타" 계산이 MAC-IRL의 검증 절이 SPR만큼
> 간결할 수 있다고 가정했으나 불가능하다. MAC-IRL은 절차가 5개이고 파라미터가
> 더 많으며(train/test 일수, block length 20, 10,000 resample, FDR),
> 사전 식별 규칙·RMSE·"forecast 아님/p-value 아님" 단서가 §4의 의존 대상이라
> 뺄 수 없다. 실측 절별: 도입 84 / 3.1 165 / 3.2 619 / 3.3 371 / 3.4 756.
>
> **시험 splice 컴파일 결과** (현행 판과 동일 조건):
>
> | | 쪽수 | 8쪽 텍스트 줄수 | §5 Conclusion 위치 |
> | --- | ---: | ---: | --- |
> | 현행 | 8 | 70 | p8 |
> | 초안 v3 | 8 | **37** | **p7** |
>
> 쪽수는 8쪽 유지(참고문헌이 8쪽을 차지). 다만 8쪽 사용량이 절반으로 줄어
> **약 반 쪽의 여유** 확보. 수식 라벨은 (1)~(7)로 정상 해결, undefined 참조 0건.
>
> 추가 감축 여지 ≈150~200단어 (→ 약 1,800, −28%):
> V1의 train/test 일수 → §4.1 / 3.3 "Concavity is what makes intensity
> informative" → §1에 이미 있음 / 3.3 말미 disclaimer → §4·§5와 중복 /
> ablation의 pooling 단서 압축.

**추가 여지:** SPR은 4쪽이라 Related Work가 없어 피처 근거를 각 문단에서 처리한다.
MAC-IRL은 §2가 있으므로 근거를 §2로 넘기고 §3.2는 **정의만** 남길 수 있다.

**남는 긴장:** SPR 도입부 *"describe conditional associations in aggregate flows,
not structural preferences"* 는 복원 주장을 포기하는 문장인데 이 논문 제목은
"Recovering ... Preferences"다. 골격을 가져오면 이 충돌이 §3 안으로 들어온다.
→ 미결 #4와 동일 사안.

---

## 3. 코드 대조 — 실제 실행되는 것

`src/models/continuous.py` `forward()`:

```python
state_score = features @ weights + context @ context_main   # g_{i,t}
action      = torch.clamp(state_score, -1.0, 1.0)           # clip
reward_at_policy = state_score*action - 0.5*action.square() # ← 반환만 됨
```

`src/training/continuous_trainer.py:105-108`:

```python
mse  = F.mse_loss(model(features, context)["action"], target)
loss = mse + lambda_l1 * model.l1_penalty()
```

| 논문 수식 | 코드 | 판정 |
| --- | --- | --- |
| (1) $u_{i,t}$ | 라벨 생성 | 사용 |
| $g_{i,t}=(\beta+BC)^\top x+\alpha^\top C$ | `state_score` | 사용 |
| $\hat a=\mathrm{clip}(g,-1,1)$ | `torch.clamp` | 사용 |
| (5) MSE + $\lambda\lVert\theta\rVert_1$ | `mse + lambda_l1*l1_penalty` | 사용 |
| (2) $R_i$ | `reward_at_policy` — grep 결과 **정의 줄 1곳뿐, 소비처 없음** | 死코드 |
| $\kappa$ | 변수 없음. docstring + 하드코딩 `0.5` | 미사용 |
| $a^*=\arg\max R_i$ | 최대화 코드 없음. clamp 직접 적용 | 미사용 |
| $\pi_i$, $\sigma_i$ | **저장소 전체 0회** | 부재 |

`reward_at_policy`가 死코드인 것은 **논문 판단의 근거가 아니다**(미사용 진단값일
뿐). 다만 코드 정리 대상으로 남는다.

---

## 4. 발견된 오류

### 4.1 첨자 불일치 (식 2)

상태는 $s_{i,t}$, 목표는 $a_{i,t+1}$, 3.3 예측은 $\hat a_{i,t+1}$, 식 (5)도
$a_{i,t+1}$. 그런데 식 (2)만 $a^*_{i,t}$. 같은 대상인데 첨자가 다르다.
→ $a^*_{i,t+1}$로 통일.

### 4.2 clip이 유도되지 않음

현행 식 (2)는 $R_i$와 $a^*=\mathrm{clip}(\cdot)$을 **연결어 없이 나란히** 둔다.
3.3에서 $\hat a=\mathrm{clip}(\hat\theta^\top w,-1,1)$이 또 선언된다. 독자에게
clip은 두 번 **주장**되었을 뿐 한 번도 유도되지 않았고, 자연스러운 독해는
**clip = 링크 함수**(tanh·로지스틱 자리)가 된다. 그 독해의 귀결:

1. 보상이 장식이 된다 → "왜 tanh가 아니고 clip인가"에 답이 없다 → IRL 주장 붕괴
2. 경계 ±1의 출처가 사라진다 (실제로는 식 (1)이 비율이라 구조적으로 $[-1,1]$)
3. $\kappa$가 의미를 잃는다 (내부해 조건 $g=\kappa a$ = 한계편익·한계비용 일치)
4. **크기 해석의 근거가 사라진다** — 3.2가 이미 주장하는
   "read as magnitudes rather than signs alone"은 내부해가 최적일 때만 참

유도:

$$\partial R_i/\partial a = g-\kappa a,\quad \partial^2 R_i/\partial a^2=-\kappa<0$$

강오목 + $\mathcal{A}$ 컴팩트 → 최대해 유일 존재, 무제약 정류점
$a^\circ=g/\kappa$의 $[-1,1]$ 위 사영 = $\mathrm{clip}(g/\kappa,-1,1)$.
**clip은 이 프로그램의 해 연산자(사영)이지 고른 함수가 아니다.**

### 4.3 "convex, so it has a global solution" — 3중 비약

현행 3.5 첫 문장:
> Problem~(5) is convex, so it has a global solution that coordinate descent attains exactly.

| 주장 | 문제 |
| --- | --- |
| 볼록 → 해의 **존재** | 성립 안 함. 볼록함수는 최소해가 없을 수 있다($e^{-x}$). 존재를 주는 것은 **강제성** — $\lambda>0$의 $L_1$ 항 |
| 볼록 → 해의 **유일** | 성립 안 함. 최소해 집합은 볼록집합이지 점이 아니다. 유일성을 주는 것은 $\mathrm{rank}(W_i)=11$(완전 열계수 → 강볼록) |
| 볼록 → **좌표하강이 전역해 도달** | 성립 안 함. 일반 비평활 볼록에서 좌표하강은 정지 가능. Lasso에서 되는 이유는 벌점의 **분리가능성**(Tseng), 볼록성이 아님 |
| "**exactly**" | 좌표하강은 반복법·허용오차 종료(sklearn `tol=1e-4`). 폐형해 아님 → "to numerical tolerance" |

### 4.4 식 (5)에 clip을 넣으면 볼록성이 깨진다

논의 중 코드 정합을 위해 (5) 안에 clip을 넣자는 제안이 나왔으나 **철회**.
$h(z)=(\mathrm{clip}(z,-1,1)-a)^2$는 $a=0$일 때 $|z|\le1$에서 $z^2$,
$|z|>1$에서 $1$. $z=1$에서 좌미분 2, 우미분 0 → 기울기 감소 → **볼록 아님**
(볼록함수는 유계일 수 없는데 $h\le1$). 포화율 0은 $\hat\theta$ **근방의 국소
일치**이지 전역 볼록성이 아니다.

**정합적 서술 순서 (확정):**

1. 식 (5)는 clip **없이** 쓴다 → 볼록 · 강제적 · (완전계수 하) 유일
2. 코드가 최소화하는 것은 clip이 들어간 목적함수임을 밝힌다
3. 해에서 포화율 0임을 **관측 사실로** 보고 → 표본 위에서 두 목적함수 일치

포화율 0은 가우시안 정책과의 동치를 지키는 데 쓰지 말고, **(5)↔코드 일치**만
담당시킨다. $\pi_i$를 빼면 이 역할이 단순해진다.

**실측 근거 확인됨:** `experiments/2026-07-31/exact_lasso/saturation.csv`
— 45 split × 3 유형 = 135행 전부 `saturation_rate = 0.0` (max=min=0.0).

---

## 5. 3.5 처리 — 결론만 남기고 장치는 제거

### 등상관집합 논증이 불필요한 이유

현행은 두 갈래다: $\lambda=0$이면 $\mathrm{rank}(W_i)=11$ 확인,
$\lambda>0$이면 KKT 벡터 $c_i$·등상관집합 $E_i$·$\mathrm{rank}(W_{i,E_i})=|E_i|$
확인. 그런데

- 원고 스스로 *"full column rank of $W_i$ implies the latter for every subset"*
  이라 적고, 실제로 45 split 전부 rank 11 → **두 갈래가 하나로 붕괴**
- 등상관집합 기계는 $p>n$ 또는 계수 부족일 때의 일반형. 여기는 $n\approx770$,
  $p=11$, 조건수 6.94(유한 → 이미 완전계수 함의)
- $\lambda=0$ 갈래는 **일어나지 않는 경우**. λ=0.005 공통이고 V4도 $L_2$ 격자만
  (SPR V4에는 무벌점이 있으나 본 논문에는 없음)

### 남겨야 하는 이유

논문은 "Lasso가 기관 모멘텀 계수를 45개 중 55.6%, underwater를 77.8%에서 정확히
0으로 놓는다"를 해석한다. 해가 유일하지 않으면 **"계수가 0이다"라는 진술 자체가
무의미**하다(같은 최적값의 다른 해에서 0이 아닐 수 있음). 즉 유일성은 부호 해석이
아니라 **정확한 0 주장**을 떠받친다. 이 연결을 명시하면 한 문장이 정당화된다.

> Across all 45 splits and all three types, $W_i$ has full column rank 11
> (worst-case condition number 6.94), so the squared-error term is strictly
> convex and the minimizer of (5) is unique. Uniqueness is what licenses
> reading an exactly-zero coefficient as a property of the estimator rather
> than of one arbitrary solution among many.

374단어 → 약 50단어.

### 행선지 표

| 현행 3.5 내용 | 처리 |
| --- | --- |
| 볼록성 → 전역해 → 좌표하강 "exactly" | **삭제** (§4.3) |
| $\lambda=0$ 분기, KKT/등상관집합 | **삭제** (미발생 · 완전계수에 흡수) |
| rank 11 · 조건수 6.94 · 유일성 | **유지**, 정확한 0 주장과 연결 |
| $\lambda=0.005$ 공통 + 비교가능성 논거 | **유지** (2문장) |
| Adam seed 42 / 75 epochs / 유형별 batch·lr | **4장 또는 각주** — 방법 아님 |
| 좌표하강 vs Adam 0.001 일치 감사 | **각주 또는 삭제** (근거: `experiments/2026-07-31/1329_정확Lasso_대조.md`) |

---

## 6. 삭제 대상 — 설명이 아니라 변명인 것

- 3.2: concavity 옹호 문단 (~80단어). 강오목성이 유일성·내부해를 준다는 것이
  §4.2 유도에 드러나므로 불필요
- 3.3: "We state this reduction rather than obscure it",
  "our contribution is the validation protocol ..., not a new estimator",
  "Two limits ... we state them here rather than deferring them"
- 3.4 말미: "The expectations above are fixed before estimation, so
  Section 4 reads ... rather than rationalizing them afterwards"
- 3.6 말미: "Two earlier claims were withdrawn on these grounds"
- 3.1 데이터 문단 (삼성전자·973일·2022-01-06~2025-12-29) — §4.1과 중복
- 3.1 피처 선요약 ("20-day momentum, own lagged flow, ...") — 3.4와 중복

SPR은 이 역할을 §1 도입 한 문장(*"Its coefficients describe conditional
associations in aggregate flows, not structural preferences"*) + §4 Limitations
로 처리한다.

---

## 7. 확정 문안 초안

### 3.3 Reward and policy (도입부)

> Each type chooses its next-day action from the feasible set
> $\mathcal{A}=[-1,1]$, the range of the normalized net-buy in (1). The reward
> of action $a$ in state $s_{i,t}$ is
> $$R_i(a\mid s_{i,t})=g_{i,t}a-\tfrac12\kappa a^2,\qquad \kappa>0,$$
> with $g_{i,t}$ the marginal reward of the first unit traded and $\kappa$ the
> curvature of the quadratic cost. $R_i$ is strictly concave and $\mathcal{A}$
> compact, so the optimal action exists, is unique, and is the projection of
> the unconstrained optimum onto $\mathcal{A}$:
> $$a^{*}_{i,t+1}=\arg\max_{a\in\mathcal{A}}R_i(a\mid s_{i,t})
>   =\operatorname{clip}(g_{i,t}/\kappa,-1,1).$$
> The clip is therefore the boundary of $\mathcal{A}$, not a link function
> chosen for convenience. Only $g_{i,t}/\kappa$ is identified from observed
> actions, so we set $\kappa=1$; magnitudes are comparable across types because
> the same normalization is imposed on all three.

### 3.4 추정 — 정책 없이 추정기로 연결

$a^*$가 정의되면 회귀의 추정 대상에 이름이 붙는다. 보상에서 추정기로 가는 다리는
$a^*$이지 $\pi$가 아니다:

$$\hat\theta_i=\arg\min_\theta\ \frac{1}{|\mathcal{D}_i|}\sum_{t\in\mathcal{D}_i}
\bigl(a^{*}_{i,t+1}(\theta)-a_{i,t+1}\bigr)^2+\lambda_i\lVert\theta\rVert_1$$

= 모형 최적행동과 관측행동의 제곱거리 최소화(행동 복제). **분포 가정 불필요** →
가우시안 정책을 뺄 수 있는 근거. 단 §4.4에 따라 본문 식은 clip 없는 형태로 쓰고
포화율 0을 별도 진술한다.

---

## 8. §1·§4·§5 대조 (2026-07-31 추가)

### 8.1 프레이밍 충돌 — 거의 없음 (앞선 우려는 과장, 정정)

논문은 이미 SPR과 같은 입장을 일관되게 취한다:

| 위치 | 문장 |
| --- | --- |
| 초록 | "reveals a latent reward" / "We make no claim of return predictability" |
| §1 | "recover ... **a description of** what each investor type responds to" |
| §4 Validation and scope | "All statements remain **conditional associations** in one stock: market clearing, common shocks, beliefs, institutional mandates, and within-category heterogeneity are not separately identified" |
| §5 | "it **cannot separate structural preferences** from market clearing, shared information, or aggregation" |
| §5 | "$\beta_i$ would carry the structural reading that the myopic estimates here **deliberately withhold**" |

→ SPR 도입부 disclaimer를 §3에 넣는 것은 충돌이 아니라 **§4·§5와의 정합**.
제목 "Recovering ... Preferences"만 가장 느슨하나 초록 첫 문장이 곧바로 한정.
**미결 #4 대체로 해소.**

### 8.2 삭제 금지 목록 — 다른 장이 §3에 의존하는 것

| 다른 장의 문장 | §3에서 반드시 살아남아야 하는 것 |
| --- | --- |
| §1 "A concave reward makes the optimal action graded, so trading intensity is derived rather than assumed" | $R_i$ + argmax 유도 (D7 확인) |
| §1 기여② "explicit disclosure of what the model fails to identify" · §4 "preregistered rule" ×2 | identified/unidentified 사전 기준 |
| 초록·§1 "construct-validity checks against independently established behavioral regularities" | **사전 부호 기대** |
| §4 "V3 shows that persistence reverses in one of three temporal windows" | V3 3창 + "세 창은 약하다" 단서 |
| §5 "the same ... **normalization**" | $\kappa=1$ |

→ **미결 #2 해소**: 부호 기대는 선택이 아니라 **필수**. 초록이 construct-validity를
기여로 내세우므로 §3에 사전 기대가 없으면 그 주장이 공중에 뜬다.
배치는 §3.2 말미 한 문단(또는 표)으로 통합하되 **삭제 불가**.

### 8.3 가산 방식의 최대 위험 — SPR 수치 오염

SPR과 MAC-IRL은 **2번 피처가 다르다**(SPR $x^{cross}$ 타 유형 시차 /
MAC-IRL $x^{per}$ 자기 시차). 결과 전체가 다르다.

| | SPR | MAC-IRL |
| --- | ---: | ---: |
| foreign momentum | +0.0502 | **+0.0435** |
| retail / individual momentum | −0.0383 | **−0.0356** |
| 2번 피처 부호 | lagged cross-flow, 세 유형 **전부 음수** | persistence, 세 유형 **전부 양수** (+0.0505 / +0.0462 / +0.0102) |
| 기관 결론 | "no stable common behavioral weight" (완전 null) | persistence·KOSPI **식별됨** (제한적 구조) |
| 조건수 | 6.86 | 6.94 |
| V2 부트스트랩 | 1,000회 | 200회 |
| 유형 명칭 | individual | **retail** |

**SPR 본문의 수치·결론은 전부 stale.** 골격과 문장 구조만 이식하고 숫자는 한 개도
옮기지 않는다. 용어도 individual → retail 변환 필요.

### 8.4 새로 발견된 결함

1. **용어 불일치.** §1 "shared specification" vs §5 "**symmetric** recovery design".
   D2가 Symmetric을 폐기했는데 §5에 잔존 → §5 수정 필요.
2. **삭제 예정분은 이동이 아니라 삭제로 충분.** §3.3의 반사실 전이 부인은 §5의
   "cannot separate structural preferences from market clearing, shared
   information, or aggregation"이 이미 덮는다. §1에도 있는
   "We state this rather than obscure it"은 §3.3과 중복.
   → **미결 #1 해소**: 행선지 탐색 불필요, 삭제. 단 Adam 설정은 §4.1 각주로
   (§4.1이 이미 "The specification, penalty, and CPCV design are those of
   Section~3"이라 자연스러움).
3. **초록의 검증 목록이 V1–V4와 매핑되지 않음.** 초록은 CPCV · block bootstrap ·
   walk-forward · construct-validity · null disclosure를 나열 —
   **V4(벌점 대체)와 ablation이 누락**. §3이 5개 절차를 명세하는데 초록은 4개.
   → 초록 수정 검토 대상.

---

## 9. 최종본 확인사항 (지금 처리 안 함)

§3 재작성으로 §3.1의 데이터 문단을 삭제했다. §4.1 첫 문장이 그 문단을 참조하고 있다:

> We evaluate the common specification in a single-stock controlled setting using
> Samsung Electronics (005930), on the 973 state days **defined in
> Section~\ref{sec:method-setup}**.

`\ref`는 절 번호만 찍으므로 LaTeX 경고가 나지 않는다. 다만 §4.1이 이미
**Samsung Electronics (005930)**, **973 state days**, **single-stock controlled
setting** 을 모두 갖고 있으므로 실제로 빠진 것은 **표본 시점 하나**뿐이다.

논문 전체에서 표본 시점이 나오는 곳: §3.4 V3의 "testing on 2023, 2024, and 2025"
와 Table 2 캡션의 "2023--2025". 끝이 2025인 것은 추론 가능하나 **시작 시점은
어디에도 없다.**

최종본에서 필요하다고 판단되면 §4.1 첫 문장만 고치면 된다:

```diff
- on the 973 state days defined in Section~\ref{sec:method-setup}.
+ on 973 state days running from 6~January~2022 to 29~December~2025.
```

30 → 33단어. §3은 건드리지 않는다.

불필요하다고 판단한 것 (원래 §3.1 문단에 있었으나 복원하지 않기로 함):

| 항목 | 사유 |
| --- | --- |
| 2021--2025 원자료 창, 1,225 → 973 산술 | §3.2가 252일 창을 정의하므로 유도 가능 |
| 252일 warm-up 설명 | 동일 |
| single-stock 상충 문장 | §5가 일반화를 후속 과제로 다룸 |

---

## 10. 미결

| # | 항목 | 상태 |
| --- | --- | --- |
| ~~1~~ | ~~3장에서 빼는 내용의 행선지~~ | **해소** (§8.4-2) — 삭제로 충분. Adam만 §4.1 각주 |
| ~~2~~ | ~~피처별 부호 기대 처리~~ | **해소** (§8.2) — 필수 유지. 배치는 §3.2 말미 통합 |
| 3 | 산출물 형식: tex 직접 교체 vs 초안 파일 선행 | **미정** |
| ~~4~~ | ~~제목·§1·§4·§5 정합성~~ | **대체로 해소** (§8.1). 잔여: §5 "symmetric" → "shared-specification" |
| 5 | `reward_at_policy` 死코드 제거 | 코드 정리 대상 |
| 6 | SPR 1.1 절 제목 재명명 ("Behavior Forecasting Problem"은 IRL 프레임과 충돌) | **미정** |
| ~~7~~ | ~~SPR 문안 재사용 가부~~ | **해소** — 같은 논문 공저자 내부 초안 |
| 8 | 초록의 검증 목록에 V4·ablation 누락 (§8.4-3) | **신규** |

**다음 액션:** D9(가산 방식)로 §3 집필. 미결 3·6만 확정하면 착수 가능.
§3 완료 후 별건으로 미결 4 잔여(§5 용어)·8(초록) 처리.
현재 원고 미수정 상태 유지.
