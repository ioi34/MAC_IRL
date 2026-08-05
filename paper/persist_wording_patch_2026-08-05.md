# persist 시차 — 표현만 수정하는 안 (수치 재실행 없음)

대상 파일: `paper/mac_irl_icaif26_full_en.tex`
전제: `x^{per}_{i,t} = a_{i,t-1}` 을 그대로 두고, 타깃 `a_{i,t+1}` 대비 lag-2 라는 사실과
피처 간 시점 정렬 차이를 본문에 드러낸다. 표·그림의 수치는 바뀌지 않는다.

고쳐야 할 곳은 세 군데다. (1)이 핵심이고 (2)(3)은 (1)과 어긋나지 않게 맞추는 정리다.

---

## (1) §3.2 Own flow persistence — 456~465행

**현행**

```latex
\textbf{Own flow persistence.} $x^{\mathrm{per}}_{i,t} = a_{i,t-1}$, the
previous day's action on the same own-denominator definition
as~\eqref{eq:action}, so the coefficient is a first-order autoregression
on behaviour.
```

**수정안**

```latex
\textbf{Own flow persistence.} $x^{\mathrm{per}}_{i,t} = a_{i,t-1}$, the
action one day before the state date, on the same own-denominator
definition as~\eqref{eq:action}. Because the target is $a_{i,t+1}$, the
coefficient is a second-order rather than a first-order autoregression on
behaviour: the intervening action $a_{i,t}$ is public before day $t+1$
trading begins and could have been used, and excluding it makes this
feature a deliberately conservative measure of persistence. It therefore
lags the other two features, which are both evaluated at day~$t$, and the
coefficient should be read as a lower bound on how strongly a type
continues its own flow.
```

두 가지를 한다. `first-order` → `second-order`로 바로잡고, `a_{i,t}`를 안 쓰는 것이
누락이 아니라 선택임을 명시한다. 후자가 없으면 리뷰어가 "왜 하루를 버렸는가"를 묻는다.

`lower bound`라는 표현은 실증적으로 뒷받침된다 — `experiments/2026-08-05/1724_persist_lag_and_exact_lasso.md`
참조. `u_t`로 바꾸면 persist 계수가 세 유형 모두 커진다(기관 0.0103 → 0.0236).

---

## (2) §3.1 상태 정의 — 429행

**현행**

```latex
contains only information observable through day~$t$
```

**수정안**

```latex
contains only information observable through day~$t$; the persistence
feature uses day~$t-1$, one day short of that bound, as noted in
Section~\ref{sec:method-features}
```

현행 문장은 참이지만 "through day t"가 곧 "day t까지 전부 쓴다"로 읽힌다.
실제로는 피처마다 시점이 다르므로 여기서 한 번 짚어 둔다.

---

## (3) 그림 1 — 383행 및 f2 셀

라벨 `$a_{i,t-1}$`은 이미 정확하므로 그대로 둔다. 다만 f2 셀 문구가
`own lagged\\ action` 이라 lag의 크기가 안 보인다.

```latex
\node[cell=11mm] (f2) at ([yshift=-11mm]feat.north)
  {$x^{\mathrm{per}}_{i,t}$\\ own action\\ at $t-1$};
```

캡션에도 한 문장 덧붙인다.

```latex
Yesterday's action re-enters as the persistence feature (dashed), so that
feature lags the target by two days while the other two lag it by one.
```

---

## 남는 위험

이 경로는 수치를 지키는 대신 다음을 감수한다.

1. **기관 결과의 해석이 흔들린다.** §4.4는 기관에 구별되는 성향이 없다고 결론짓지만,
   시차를 고치면 기관이 평균-행동 벤치마크를 넘는다(OOS R² -0.0036 → +0.0231).
   "conservative measure"라고 써 두면 성실하긴 하나, 심사 과정에서 재실행을 요구받을
   경우 결론 자체가 바뀔 수 있는 상태로 남는다.
2. **`herd` 피처도 같은 `.shift(1)`을 갖는다.** 지금 canonical run은 persist를 쓰지만
   `runs/continuous_reward3_lambda_unified/`는 herd를 쓴다. 어느 쪽이든 동일한 정정 문구가
   필요하다.
3. **성능 수치 전반이 낮게 보고된다.** foreign corr 0.310 vs 0.364,
   retail 0.242 vs 0.312. 전이 실험의 감쇠 해석(§4.3)도 이 낮은 기준선 위에서 쓰여 있다.
