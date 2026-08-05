# §4·§5 담당자 인계 메모 — 2026-07-31

동료판 §4(`4page-en-writing-skill (1).tex`)를 원고에 병합해 검증한 결과다.
**§4 본문·수치·판정은 손대지 않았다.** 아래는 §4/§5 쪽에서 처리해야 할 항목만 모았다.

---

## 1. 필수 — `\label{sec:discussion}` 이 사라져 참조가 깨진다

§3.2(487행)가 `\ref{sec:discussion}`을 쓰는데 동료판 §5는 `sec:conclusion`만 정의한다.
빌드에서 `Reference 'sec:discussion' undefined`.

**처리 완료(§3 쪽):** §3.2의 참조를 `\ref{sec:conclusion}`으로 바꿨다.
§4/§5 쪽에서 할 일은 **없다.** 라벨을 되살리지 말 것 — 한 절에 라벨 두 개는 불필요하다.

## 2. 필수 — §5에 한 문장 추가 (B안)

§3.2와 Intro가 §5에서 structural recovery를 다룬다고 **두 번** 약속하는데
동료판 §5는 다루지 않는다. §3.3이 "the estimator is a Lasso"라고 선언한 뒤
"그럼 왜 IRL인가"에 답하는 유일한 문장이 §3.2의 이 약속이라, 비워두면 논증이 끊긴다.

§5 future-work 문단의 `...and to finer institutional categories.` **바로 뒤에** 삽입:

```latex
A longer step is to relax the one-step horizon: under a genuine
planning horizon the reward is no longer informationally equivalent to the
conditional best response, and $\beta_i$ would carry the structural reading
that the myopic estimates here deliberately withhold.
```

새 주장이 아니다 — §3.3이 이미 "Because the horizon is one step, the recovered
reward is informationally equivalent to the conditional best response"라고 썼고,
이 문장은 그 조건문의 뒷면일 뿐이다. 추가 근거·추가 실험 불필요.

## 3. 확인 요망 — "preregistered" 라는 단어

동료판 §4는 식별 규칙을 "the **preregistered** rule"이라 부르고, 표 캡션은
"at least **90\%** V1 sign consistency"라고 한다.

§3.6은 원래 100%(`sign is consistent across the 45 splits`)였고, **오늘 90%로 맞췄다.**

- 90%와 100%가 갈리는 계수는 **외국인 FX $\alpha$ 하나뿐**이다(V1 97.8%, V2 Yes).
  90% 기준에서만 identified이고, 동료판 Table 3·"Validation and scope"가
  "their four direct context effects"라고 쓴 것이 여기에 의존한다.
- ⚠️ **90%가 오늘 정해진 값이라면 "preregistered"라고 쓰면 안 된다.**
  리뷰어가 가장 세게 때린 지점이 "사전등록했다고 해놓고 안 지킨다"라서,
  사후 임계값을 사전등록이라 부르면 같은 지적을 다시 받는다.
  → 원래 정해져 있던 값인지 확인하고, 아니라면 "preregistered" 표현을 조정할 것.

## 4. 사실 오류 — §4.2 수렴 문단

동료판 §4 114행:

> The 75-epoch fits also make incomplete optimization an unlikely explanation for
> the institutional result. The last epoch contributes only 1.6\%, 0.4\%, and 0.1\%...

`experiments/2026-07-31/1329_정확Lasso_대조.md` 결과와 충돌한다. 동일 목적함수를
정확 Lasso(coordinate descent)로 다시 풀면 **외국인은 split별 최대 0.0167**
(계수 크기 0.033의 절반) 벗어나 있다. Adam은 split 단위 전역해에 도달하지 못했다.

기관에 한정하면 이 문장은 맞다(기관 최대 차이 0.00055). 하지만 문장이 세 유형을
싸잡아 "convergence는 문제 아님"으로 읽히므로 수정이 필요하다.

**권장:** 이 문단(3줄)을 삭제하고 §3.5의 정확 솔버 문장으로 대체.
페이지도 5줄 확보된다. §3.5 수정은 §3 쪽에서 처리 예정.

## 5. 참고 — 병합·레이아웃 검증 결과

| 상태 | 페이지 |
| --- | ---: |
| 동료판 §4 병합, 전폭 플로트 그대로 | 10p |
| + 단단 레이아웃 패치 | **8p** (마지막 쪽 91/118줄, 여유 ≈0.23p) |

패치는 `scripts/apply_1col_layout.py`가 자동 처리한다. 신·구 §4 라벨 양쪽을 지원하므로
§4를 더 고쳐도 그대로 다시 돌리면 된다.

    python3 scripts/apply_1col_layout.py <원고.tex> -o /tmp/patched.tex

패치가 하는 일: Table `tab:context-main`의 $\alpha$ 6행을 `tab:reward-weights`
하단에 흡수, Figure `fig:context-effects`를 `fig:reward-validation`에 흡수(2패널),
두 그림을 단단 `figure`로 전환. **본문 문장은 건드리지 않는다.**

## 6. $\alpha$ 의 V3 는 값이 없다 — `---` 로 둘 것

병합표의 $\alpha$ 6행 V3 열은 전부 `---`다. 추측값을 넣지 말 것.
`scripts/train_continuous_walk_forward.py`가 `context_main_effect`를 모델에 전달하지
않아 **walk-forward는 $\alpha$를 아예 추정하지 않는다.**
코드는 고쳤으나 재실행 전이다 — `experiments/2026-07-31/1323_walkforward_alpha_수정.md`.

동료판 §4가 Table 3에 V3 열을 두지 않은 것은 결과적으로 옳은 선택이다.

⚠️ 재실행하면 **$\beta$의 V3 값도 바뀔 수 있다.** 현재 walk-forward는 $\alpha$ 없는
축소 사양으로 재적합한 결과라, §4.2의 V3 문장 전체가 그 위에 서 있다.
재실행 후 기관 persist의 1/3 창 반전이 유지되는지 반드시 확인할 것.
