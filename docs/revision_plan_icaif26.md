# ICAIF '26 리비전 점검 — 2026-07-31

> 대상: `acmart-primary/mac_irl_icaif26.tex` (= `paper/mac_irl_icaif26.tex`, 두 파일 동일)
> 빌드: `MAC_IRL/mac_irl_icaif26.pdf` (2026-07-31 01:11, **10 페이지**)
> canonical run: `runs/continuous_reward3_persist_epochs75/`
> 검증 스위트: `runs/continuous_reward3_persist_epochs75_validation/` (α 버그 수정 후 2026-07-30 16:25 재실행분)
> 마감: 2026-08-02 AOE (약 48시간)

---

## 0. 요약

수치는 깨끗하다. 원고 §4의 모든 값을 canonical run CSV와 대조해 **불일치 0건**이다.
문제는 수치가 아니라 두 가지다.

1. **페이지 초과 (10p / 한도 8p)** — 데스크 리젝 사유. 리뷰 지적사항보다 우선한다.
2. **사전등록 식별 기준과 실제 판정의 불일치** — 리뷰어 단점 1. 그리고 리뷰어가 파악한 것보다
   한 단계 더 나쁘다 (§2.1 참조).

리뷰어가 제시한 개정 항목 4개 중 (d)는 이미 해결돼 있고, (a)(b)(c)는 미해결이다.

---

## 1. 실험 상태

### 1.1 수치 대조 결과 — 전부 일치

| 원고 위치 | 주장 | 출처 CSV | 판정 |
| --- | --- | --- | :---: |
| Table 1 | OOS R² +0.130 / −0.003 / +0.107 | `verify_paper_numbers_epochs75/out_of_sample_r2_summary.csv` (+0.1301 / −0.0033 / +0.1068) | ✅ |
| §4.1 | AR(1) 0.401 / 0.149 / 0.345 | `action_autocorrelation.csv` | ✅ |
| §3.5, §4.2 | rank 11, 최악 cond 6.94 | `design_matrix_uniqueness.csv` (135조합 결손 0) | ✅ |
| Table 2 | β 9개 값·V1 일관성 | `ablation/baseline/reward_weights_summary.csv` | ✅ |
| Table 2 | V2 (0 배제 여부) 9개 | `weight_bootstrap/bootstrap_reward_weights_summary.csv` | ✅ |
| Table 3 | α 6개 평균·95% CI | `weight_bootstrap/bootstrap_context_main_weights_summary.csv` | ✅ |
| §4.2 | ablation q=0.0007 / 0.0014 / 0.0322 (저하) | `analysis/ablation_paired_bootstrap.csv` | ✅ |
| §4.2 | 기관 momentum 제거 q=0.0122 / 0.0231 (개선) | 〃 (`q_improvement`) | ✅ |
| §4.2 | 마지막 epoch 손실기여 1.6% / 0.4% / 0.1% | `convergence_summary.csv` (0.0157 / 0.0043 / 0.0014) | ✅ |

**주의 한 가지.** 개인 underwater의 V2는 `1536_persist_epochs75_검증스위트.md`에
"[0.0100, 0.0695] 배제"로 기록돼 있으나, 그 뒤 16:25 α 버그 수정 재실행으로
**[−0.0028, 0.0581] 미배제**로 바뀌었다. 원고 Table 2의 "No"가 최신 값 기준으로 맞다.
실험 노트 1536은 재실행 전 시점 기록이므로 그대로 인용하면 안 된다.

### 1.2 검증 스위트 완결성

- CPCV 45 split, bootstrap 200, walk-forward 3창, ridge 15단계, ablation 8종 — 전부 완주.
- **미실행/불가**: `scripts/analyze_continuous_reward_validation.py`의 마지막 단계
  (momentum↔relative 상관 분리)는 5특징 구성 전용 하드코딩이라 3특징 persist 구성에서
  `FileNotFoundError`. 기존 한계이며 원고가 이 분석에 의존하지 않는다.
- **존재하지 않는 실험**: 정확 Lasso 솔버(coordinate descent / LARS) 대조. 저장소 전체에
  sklearn `Lasso` 호출이 없다. 리뷰 개정항목 (b)의 근거가 될 산출물이 **없다.**

### 1.3 저장소 상태

- `paper/mac_irl_icaif26.tex` 미커밋 (+342 / −39). `.claude/` untracked.
- `acmart-primary/mac_irl_icaif26.pdf`는 2026-07-27 빌드로 **제목이 옛 버전**
  ("Same Signals, Different **Rewards**"). 최신 빌드는 `MAC_IRL/mac_irl_icaif26.pdf`.
  제출 직전 어느 PDF를 올리는지 확인 필요.
- §4 주석이 `docs/revision_plan_icaif26.md` appendix A를 인용하는데 그 파일이 없었다
  (이 문서가 그 자리를 대신한다).

---

## 2. 리뷰 지적사항별 상태

### 2.1 (a) 식별 기준 불일치 — **미해결, 리뷰어 지적보다 심각** 🔴

§3.6의 사전등록 규칙: `V1 부호 100% AND V2 구간이 0 배제` → identified, 아니면 unidentified.
같은 문단에서 "V3는 식별 판정이 아니라 시간 안정성 증거로만 보고한다"고 명시한다.

그런데 규칙을 그대로 적용하면 **기관에 identified 계수가 두 개 나온다**:

| 계수 | V1 | V2 | 규칙상 판정 | 원고의 실제 처리 | 강등 근거 |
| --- | ---: | --- | --- | --- | --- |
| 기관 persist β | **100%** | 배제 [0.00007, 0.0201] | **identified** | Boundary (Table 2 각주 a, §4.2) | V3 1창 반전 + 하한 근접 + R² 음수 |
| 기관 KOSPI α | **100%** | 배제 [−0.0119, −0.0002] | **identified** | Boundary (Fig 4 캡션) | "구간이 0에 근접" |

리뷰어는 기관 KOSPI α의 V1을 "‒"(미보고)로 적었다. 실제로는
`context_main_weights_summary.csv`에서 **direction_consistency = 1.0 (45/45)** 이다.
즉 이 계수는 사전등록 기준을 **완전히** 통과한 상태에서 명문화되지 않은 판단으로 강등됐다.
Table 3은 α의 V1을 아예 싣지 않아 독자가 이 사실을 확인할 수 없다 — 리뷰어가 "‒"로
쓴 이유도 이것이다. 지금은 은폐로 읽힐 수 있는 구조다.

파생 문제: Abstract "Institutional flow shows no identified preference"는 논문 자신의
규칙 기준으로 **거짓**이다 (리뷰 개정항목 (c)).

**권고 (택1)**

- **A안 (정직·저비용).** §3.6을 3단으로 명문화:
  `identified` = V1 100% AND V2 배제 AND CI 하한/상한의 0으로부터의 거리가
  계수 크기의 X% 이상; `boundary` = V1·V2는 통과하나 위 정량 임계 미달 또는 V3 반전;
  `unidentified` = 그 외. 임계값(예: |CI 최근접단| / |평균| ≥ 0.05)을 숫자로 박고,
  Table 2·3에 판정 열을 추가한다. Abstract는
  "no preference identified at the main tier; two boundary-tier coefficients are reported"로 수정.
- **B안 (규칙 준수).** 강등을 철회하고 기관 persist·KOSPI α를 identified로 보고하되,
  §4.3에서 "identified되었으나 경제적 크기가 외국인·개인의 1/5 이하이고 R²는
  벤치마크 미달"이라고 해석 수준에서 제한. Abstract도 이에 맞춰 재서술.

A안이 리뷰어 요청 (a)+(c)를 동시에 닫는다. Table 3에 V1 열 추가는 필수.

### 2.2 (b) 볼록 문제에 SGD — **미해결, 원고 내부 모순** 🟠

- §3.5 첫 문장: "Problem (5) is convex and **coordinate descent** attains a global solution."
- 같은 절 뒤: "Fitting uses **Adam** with seed 42 and 75 epochs..."

coordinate descent를 언급만 하고 쓰지 않는다. 현재 텍스트는 리뷰어 지적을 강화한다
("솔버를 알고 있었는데 왜 안 썼나").

**권고.** 11차원 Lasso 45 split × 3유형 = 135 fit. 정확 솔버로 수 초다.
`λ_paper = 0.005`, 목적함수가 `(1/n)‖·‖² + λ‖θ‖₁`이므로 sklearn `Lasso(alpha=λ/2=0.0025,
fit_intercept=False)`로 대응시킨 뒤, 최대 계수 차이 한 줄을 §3.5에 넣는다.
예: "An exact coordinate-descent solve on the same 45 splits reproduces every coefficient
to within X×10⁻⁴, so the Adam fits are at the global optimum."
이러면 §4.2의 epoch별 손실기여 문단(3줄)을 지워도 되고, **페이지도 줄어든다.**

### 2.3 (c) Abstract ↔ Table 2 정합 — **(a)와 함께 처리**

### 2.4 (d) 허딩 범위 축소 — **✅ 이미 해결됨**

Abstract는 세 규정성을 "momentum / disposition / **persistence in a type's own flow**"로
쓰고 있고, §1(140–143행)과 §3.4가 "own-following 성분만 쓰며 herding이라 부르지 않는다"고
명시한다. 리뷰가 본 판본 이후 수정된 것으로 보인다. **추가 작업 불필요** —
다만 리뷰 대응 레터에 "이미 반영됨"으로 짚어두면 좋다.

### 2.5 단점 3 (단일 종목) / 단점 4 (기관 세분화) — 48시간 내 불가

§5가 이미 향후 과제로 정직하게 인정하고 있다. 데이터가 있어도 재실행·재검증 시간이 없다.
**손대지 말 것.**

### 2.6 단점 5 (경제적 크기 미번역) — 저비용, 페이지 여유가 생기면

β=0.0435의 의미를 원화로 환산한 한 문장. 계산: 표준화 1σ 모멘텀 증가 →
외국인 action +0.0435 → 해당 유형 평균 총거래대금 × 0.0435 원. 데이터에 있으므로 계산 가능.
Almgren–Chriss 연결은 관련연구 한 문장으로 충분하나 페이지 여유가 없으면 생략.

### 2.7 명료성 지적들 — 저비용

| 지적 | 조치 | 비용 |
| --- | --- | --- |
| κ 정규화 순서 역전 | 식(2) 앞에 "with κ normalized to one (only g/κ is identified)" 한 구절 이동 | 1줄 |
| 기호 충돌 g_{i,t} vs g_i | §3.5의 Lasso 유일성용 그래디언트를 `c_i`로 개명 — **이미 c_i로 쓰고 있음, 확인 결과 충돌 없음** | 0 |
| 식(6) 과압축 | 식 앞에 직관 한 줄 추가 | 1줄 |
| "Boundary case" 미정의 | §2.1 A안이 해결 | — |

※ 기호 충돌은 리뷰가 본 판본의 문제이고 현재 tex는 `c_i`를 쓴다. 확인 완료.

---

## 3. 페이지 초과 — 최우선 🔴

현재 **10 페이지**, 헤더 명시 한도 **8 페이지 (참고문헌 포함)**.
(제출 전 ICAIF '26 CFP에서 한도 재확인 필요.)

원인은 본문 분량이 아니라 **전폭(full-width) 플로트 과다**다:

| 플로트 | 종류 | 폭 | 높이 |
| --- | --- | --- | --- |
| Fig 1 (파이프라인, TikZ) | `figure*` | 2단 | ~27mm + 캡션 |
| Table 1 (CPCV 성능) | `table` | 1단 | 작음 |
| Fig 2 actual_vs_pred | `figure*` | 2단 | **108mm** |
| Table 2 (β) | `table*` | 2단 | 중간 |
| Fig 3 beta_stability | `figure*` | 2단 | **108mm** |
| Table 3 (α) | `table*` | 2단 | 작음 |
| Fig 4 alpha_context | `figure*` | 2단 | 54mm |

전폭 플로트 6개가 밀려 p7·p8·p10이 거의 플로트 전용 페이지가 됐다.

### 3.1 실측 결과 (2026-07-31)

원본을 건드리지 않고 사본으로 변형 빌드해 측정했다. 샌드박스 빌드의 base가
Tony 빌드와 동일한 **10p**로 나와 캘리브레이션 확인됨
(microtype expansion만 끔 — 전 변형에 동일 적용).

| 변형 | 내용 | 결과 |
| --- | --- | ---: |
| base | 현재 원본 | **10p** |
| B | Fig 4(α context)만 제거 | 9p |
| C | Table 3(α)만 제거 | 9p |
| D | Fig 2를 단단 70mm로 | 9p |
| E | Fig 3을 단단 70mm로 | 9p |
| **F** | **그림 4개 전부 제거 (텍스트+표 바닥)** | **8p** (마지막 쪽 22줄만 = 실질 7.2p) |
| G | B+C+D+E 동시 적용 | 8p (마지막 쪽 68줄) |
| **G2** | **실제 통합안** (아래) | **8p (마지막 쪽 77/118줄 → 여유 ≈0.35p)** |
| H | G2 + §4.2 수렴 문단 삭제 | 8p (마지막 쪽 72줄, +5줄 확보) |

측정 파일: `outputs/pagebudget/` (세션 스크래치, 영구 보관 아님)

### 3.2 "이미지를 먼저 다 지울까?" → 아니오

F가 답이다. **그림을 전부 지워도 8페이지**이고 여유는 0.8쪽뿐이다.
즉 텍스트+표만으로 이미 한도의 90%를 쓰고 있어서, 애초에 전폭 플로트 4개가
들어갈 자리가 없었다. 지웠다 나중에 넣으면 반드시 다시 초과한다.
게다가 acmart 플로트는 지우면 페이지 분할이 전면 재배치돼 예산이 되돌아오지 않는다.
**지우지 말고 통합·축소로 간다.**

### 3.3 확정 감축안 (= G2, 실측 8p)

1. **Table 3을 Table 2에 흡수.** α 6행을 `\midrule` + 소제목행으로 Table 2 하단에 붙이고
   `table*` 하나로 만든다. §2.1이 요구하는 **α의 V1 열이 여기서 자동 해결**된다
   (외국인 KOSPI 100% / FX 97.8%, 기관 KOSPI **100%** / FX 97.8%, 개인 100% / 100% —
   출처 `runs/continuous_reward3_persist_epochs75/context_main_weights_summary.csv`).
2. **Fig 4를 Fig 3에 흡수.** β 패널 + α 패널의 2패널 그림 하나로. 단단 `figure`, 높이 90mm.
3. **Fig 2를 단단 `figure` 70mm로.** 이미지 재생성 필요 — 3패널 세로 스택을 유지하되
   폭이 절반이 되므로 y축 라벨·범례 축소.
4. `\figorbox`의 `\includegraphics[width=\textwidth]` → `\linewidth`로 교체
   (단단 전환 시 넘침 방지). **선행 필수.**
5. (선택) §4.2 epoch별 손실기여 문단 삭제 → §2.2 정확 Lasso 한 문장으로 대체. +5줄.

**남는 여유 ≈0.35p (약 40줄).** 여기에 §3.6 3단 기준 재작성(약 12줄)과
경제적 크기 한 문장이 들어간다. 그 이상은 없다 — 새 텍스트를 넣으려면 같은 양을 빼야 한다.

### 3.4 실행 완료 (2026-07-31)

`scripts/make_paper_figures.py` — 단단 그림 2개 생성 (run CSV에서 직접).
기존 그림에는 생성 스크립트가 저장소에 없었다. 이제 재현 가능하다.

- `fig_actual_vs_pred_1col.png` (70mm)
- `fig_weights_1col.png` (90mm) — 기존 Fig 3(β) + Fig 4(α) 병합 2패널.
  마커 채움은 **V2 하나만** 사용, boundary 다이아몬드 없음 (V3 미사용 기준에 부합).

`scripts/apply_1col_layout.py` — 원고를 단단 판형으로 변환. **§4 본문·수치·판정 문장은
건드리지 않고 플로트 구조만 바꾼다.** 동료의 §4 수정이 끝난 뒤 원본에 다시 돌리면 된다.

    python3 scripts/apply_1col_layout.py paper/mac_irl_icaif26.tex -o /tmp/patched.tex

**검증 결과: 8 페이지, 마지막 쪽 70/118줄 (여유 ≈0.4p).** Overfull hbox는 25건으로
원본 24건 대비 +1 (회귀 아님).

---

## 5. 🔴 새로 발견 — walk-forward가 α를 추정하지 않는다

병합 표의 α V3 값을 검증하려다 확인했다.

`runs/continuous_reward3_persist_epochs75_validation/walk_forward/` 에
`context_main_weights.csv`가 **없다.** 원인은 2026-07-30 `1629_bootstrap_alpha_수정.md`가
진단한 바로 그 버그인데, **수정이 부트스트랩에만 적용되고 walk-forward에는 적용되지 않았다**:

    scripts/train_continuous_walk_forward.py:168
        model = ContinuousInvestorIRLModel(
            num_features=..., num_contexts=..., context_mask=context_mask,
        )                      # <- context_main_effect 인자 없음. 기본값 False.

`config_snapshot.yaml`은 `context_main_effect: true`라고 기록돼 있어 겉보기엔 정상이다.

### 파급 두 가지

1. **α에는 V3 증거가 아예 없다.** 병합 표의 α 6행 V3 열은 `---`로 두었다.
   추측값을 넣지 않도록 스크립트에 주석으로 못박아 두었다.
2. **더 심각 — V3의 β 값이 다른 사양에서 나왔다.** walk-forward는 α 파라미터 2개가
   없는 모델을 재적합한다. 즉 §4.2의 V3 문장 전체, 특히 기관 persist 강등의 근거인
   *"Institutional flow persistence reverses sign in one window"* 가
   **canonical과 다른 사양의 재적합 결과**다. 원고는 이를 밝히지 않는다.

### 선택지

- **(가) 재실행.** `train_continuous_walk_forward.py:168`에 부트스트랩과 동일한 한 줄
  (`context_main_effect=context_main_effect`)을 추가하고 walk-forward만 재실행.
  3창 × 3유형 = 9 fit이라 몇 분이면 끝난다. 그러면 α의 V3도 생기고 β의 V3도 정합해진다.
  단 **§4의 V3 수치가 바뀔 수 있다** — 동료가 지금 그 부분을 쓰고 있으므로 조율 필요.
- **(나) 미실행 + 명시.** V3가 애초에 식별 판정에 쓰이지 않는다면(현재 방향),
  §3.6에 "walk-forward는 맥락 주효과를 제외한 축소 사양으로 재적합한다"고 한 줄 명시.
  정직하지만 리뷰어가 "왜 사양이 다른가"를 물을 여지가 남는다.

동료가 마침 V3를 식별 판정에서 빼는 방향으로 §4를 고치고 있으므로 (나)로도 방어는 되지만,
(가)가 몇 분이면 끝나는 일이라 **(가) 권장**.

---

## 4. 48시간 우선순위

| 순위 | 항목 | 예상 시간 | 근거 |
| :---: | --- | --- | --- |
| 1 | 페이지 10 → 8 (플로트 통합·축소) | 3–4h | 데스크 리젝 방지 |
| 2 | §3.6 3단 기준 명문화 + Table에 판정·V1(α) 열 + Abstract 재서술 | 2h | 리뷰 (a)+(c), 핵심 방법론 정합성 |
| 3 | 정확 Lasso 솔버 대조 실행 + §3.5 한 문장 | 1h | 리뷰 (b), 원고 내부 모순 해소, 페이지도 절약 |
| 4 | κ 순서, 식(6) 직관 한 줄 | 20m | 명료성 |
| 5 | 경제적 크기 환산 한 문장 | 40m | 여유 있을 때만 |
| — | 단일 종목·기관 세분화 | — | **하지 않음** (§5 향후과제 유지) |

3번은 실행 시 `experiments/2026-07-31/HHMM_정확Lasso_대조.md`에 결과 기록 필요
(CLAUDE.md §5).
