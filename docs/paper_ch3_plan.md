# Section 3 — Method: plan and hand-off spec

> Written 2026-07-27. Primary author: **권유민**.
> Target: `acmart-primary/mac_irl_icaif26.tex` `\section{Method}`.
> This is a **plan**, not prose. The paragraph currently sitting in `\section{Method}` is raw
> material moved out of §2 — it is written in §2's voice and **must not be kept as-is**.
> Section 3.4 below says exactly how it dissolves.

---

## Locked decisions (2026-07-27)

| # | Decision | Rationale |
| --- | --- | --- |
| D1 | **The validation protocol is specified in §3; §4 reports only results.** | Matches the abstract ("we specify and apply a procedure") and contribution #2. If the procedure *is* the contribution, it belongs in Method. §4 then stays focused on tables and interpretation. |
| D2 | **3.4 presents each feature as definition → grounding → sign expectation**, one short unit per feature, with a summary table. | Keeps each feature's formula next to its justification. The alternative (definitions grouped, justification in a separate paragraph) is what §2 was doing and is the reason the reader has to backtrack. |
| D3 | **Sign expectations are stated in §3 and echoed in §4** beside the recovered coefficients. | Stating them before results is what makes §4 a test rather than a post-hoc rationalization. The echo in §4 costs about one table column. |

---

## Page budget

ICAIF 2026 allows **8 pages including figures and references, no exceptions**
([CFP](https://icaif2026.org/call-for-papers.html)). At ≈ 945 words/page in sigconf:

| Item | Estimate |
| --- | ---: |
| Title, abstract, CCS, keywords, copyright block | ~470 |
| §1 | ~500 |
| §2 | 450 |
| References (17 entries) | ~490 |
| **Available for §3 + §4 + §5** | **~5,650 (≈ 6 pages)** |

**§3 may occupy 1.5–2 pages.** There is no need to compress. Earlier framing that §2 had to
shrink for page reasons was overstated — the §2 move was justified by placement, not budget.

---

## Structure

| Sub | Title | Words | Extras | Status |
| --- | --- | ---: | --- | --- |
| 3.1 | Setup | ~180 | — | ready to write |
| 3.2 | Reward and policy | ~250 | 2 equations | ready to write |
| 3.3 | Maximum-entropy IRL and the Lasso equivalence | ~250 | derivation | ready to write |
| 3.4 | Features and sign expectations | ~250 | Table 1 | 🔴 **partially blocked (A-H1)** |
| 3.5 | Estimation | ~150 | — | ready to write |
| 3.6 | Validation protocol | ~350 | — | ready to write |

Total ≈ 1,530 words + table + equations ≈ 1.9 pages.

---

### 3.1 Setup

**Job:** establish what is observed, what is latent, and what the estimation target is.

Cover:

- Three investor types — foreign, institutional, retail — each treated as an **independent agent**. State plainly that they are estimated separately and *not* as a game; this pre-empts the "why not multi-agent IRL?" question.
- The action variable. `u_i(t) = net_buy_i(t) / trading_value(t)`, bounded in `[−1, 1]`. **Define the denominator explicitly** — it is the stock's *total* trading value, shared across types. (See A-H1: this shared denominator is exactly what broke the herd feature's interpretation. Being explicit here is now mandatory, not optional.)
- Timing. Features at `t−1`, action at `t`. State the convention once, clearly, and use it consistently — a date-alignment error already cost us one retracted claim (`experiments/2026-07-24/1745_동시점_가격영향_정렬오류_반증.md`).
- Sample. Samsung Electronics (005930), 973 trading days.

---

### 3.2 Reward and policy

**Job:** define the reward and show that trading *intensity* is derived rather than assumed.

Cover:

- Contextual linear reward gradient: `g_i = αᵢᵀC + (βᵢ + Bᵢ·C)ᵀx`, with `x` the reward features and `C` the two market contexts.
- Concave reward `R(a) = g·a − ½κa²`, `κ = 1`. Say why concave: a linear reward would push the action to a bound every day, so intensity would carry no information.
- Optimal action `a* = clip(g, −1, 1)`.
- **The point worth making explicitly:** because the reward is concave, the model predicts *how much* a type trades, not merely the direction. This is what lets us speak about magnitudes at all, and it connects directly to contribution #1 ("comparable in magnitude, not only in sign").

---

### 3.3 Maximum-entropy IRL and the Lasso equivalence

**Job:** state the equivalence openly and reframe what the contribution is.

Cover:

- IRL recovers a reward rationalizing observed behavior; non-uniqueness resolved by maximum entropy [`ziebart2008`].
- Under a **one-step (myopic) horizon**, the estimator reduces exactly to a Lasso. Show the derivation compactly.
- Then the framing sentence, which matters more than the derivation: *we state this rather than obscure it; the contribution is the validation protocol, not a new estimator.* §1 already says this — keep the wording consistent between the two.

⚠️ Do **not** oversell the IRL framing. A reviewer who spots an undisclosed Lasso equivalence will discount the whole paper; a paper that discloses it up front is simply honest about its estimator.

---

### 3.4 Features and sign expectations 🔴

**Job:** define the three reward features and the two contexts, ground each in prior work, and
commit to sign expectations before any result is shown.

#### Scope correction, 2026-07-27

The first move took the **whole** §2 paragraph, which was too aggressive. §1 names the
disposition effect and institutional herding as the motivating gap, so §2 must elaborate them
— otherwise the §1 → §2 chain breaks. The literature-establishing half has therefore been
**returned to §2** as its own block, "Behavioral regularities behind type differences."

**3.4 keeps only the mapping**: which regularity motivates which feature, and the expected
sign. Cite the regularities; do **not** re-explain them. Budget drops from ~350 to **~250
words**.

#### How the remaining material dissolves

Do **not** paste it. Distribute it:

| Sentence in the moved paragraph | Destination |
| --- | --- |
| "specified in advance rather than selected by search" | **Opening sentence of 3.4** — it is the framing claim for the whole subsection |
| Momentum grounding [`jegadeesh_titman`] | Immediately after the **momentum definition** — one clause, cite only |
| Disposition grounding [`shefrin_statman`, `odean1998`] + Grinblatt–Keloharju | Immediately after the **loss-region definition** — one clause, cite only |
| Herding grounding + the caveat | After the **third feature's definition** — 🔴 frozen, see A-H1. The *caveat* stays here (it is about our feature, not the literature); the description of what herding measures are now belongs to §2 |
| "sign expectations … carried into Section 4" | **Table 1 caption**, or the closing line of 3.4 |

Also drop these §2 artifacts: the bold run-in heading (becomes a normal subsection heading),
the phrase "with one caveat we state up front" (survey rhetoric), and the forward references
written in §2's voice.

#### Per-feature units

For each feature, in this order: **formula → one or two sentences of grounding → expected sign.**

1. **Momentum** — 20-day return. Grounding: medium-term return persistence [`jegadeesh_titman`]. Expected: **+** foreign, **−** retail, weak for institution.
2. **Loss region (underwater)** — defined relative to a decayed average cost; state the decay explicitly. Grounding: the disposition effect [`shefrin_statman`, `odean1998`], also recovered from daily Finnish trades by Grinblatt and Keloharju [`grinblatt_keloharju2001`]. Expected: **+** retail; unclear for foreign; weak for institution.
3. **Third feature** — 🔴 **BLOCKED by checklist A-H1.** The current herd feature is a near-exact affine transform of the type's own lagged flow (r ≈ −0.99), so its coefficient is not a cross-type quantity and the market-clearing justification in the moved text names the wrong mechanism. Write 3.4 with features 1 and 2 complete and leave this unit as a stub until the specification is chosen (options A–D in the checklist).

Then the **two contexts**: `kospi_return_1d`, `fx_level_z_252`. Say what each is meant to capture and note that context enters through `B` as an interaction, not as an additive term.

#### Table 1

| Feature | Definition | Grounding | Expected sign (foreign / retail / institution) |
| --- | --- | --- | --- |

Caption should carry the pre-registration point — these expectations are fixed before
estimation and §4 reads the recovered weights against them.

⚠️ If the third feature becomes `persist` (checklist Option B), it **gains** a **+**
expectation and the "no directional expectation" framing disappears. That is why the closing
sentence of the moved paragraph is frozen too.

---

### 3.5 Estimation

**Job:** make the run reproducible and justify the one choice a reviewer will probe.

Cover:

- Standardization computed on **training folds only**.
- **λ = 0.005, unified across all three types.** This is the choice to defend: a common penalty is what makes coefficient *magnitudes* comparable across types, and it is also what shows the institutional null is a real null rather than an artifact of heavier shrinkage on that type. Do not bury this.
- Optimizer: Adam [`kingma2015`] — **this citation is currently in the bibliography but uncited; 3.5 is where it belongs.**
- Seed 42; convergence criterion; confirm convergence for all three types (the institutional null depends on this).

---

### 3.6 Validation protocol

**Job (D1):** specify the procedure completely, so §4 only has to report outcomes.

Cover, in order:

- **CPCV** [`lopezdeprado2018`]: 10 folds, 2 test folds, purge 1, embargo 5 → **45 splits**. Explain *why* purging and embargo are needed with overlapping-horizon financial data; do not just state the parameters.
- **Sign consistency** across the 45 splits — define the statistic and state the threshold in advance.
- **Calendar-month block bootstrap.** ⚠️ Currently 100 resamples; checklist item A-2 raises this to 200 before submission. Report whatever number is actually run.
- **Expanding walk-forward** — 2023 / 2024 / 2025.
- **Ridge comparison** and **VIF** — collinearity diagnostics.
- **Paired block-bootstrap ablation with FDR control.**
- **A statement of what would count as failure.** This is the part that makes the protocol credible: name in advance the outcomes that would have led us to withdraw a claim. We have precedent — two claims were retracted this way (§4.7 price impact; the disposition contribution), and that record is a strength, not an embarrassment.

---

## Blocking items

| Item | Blocks | Owner |
| --- | --- | --- |
| **A-H1** — third feature specification | 3.4 unit 3, Table 1 row 3, the closing framing | decision needed from Tony |
| **A-2** — bootstrap at 200 resamples | the number quoted in 3.6 | rerun |

Everything else in §3 can be written now.

---

## Consistency checks before §3 is called done

- [ ] The Lasso-equivalence wording in 3.3 matches §1 — no drift between the two statements
- [ ] The timing convention (features `t−1`, action `t`) is stated once and never contradicted
- [ ] `kingma2015` is cited in 3.5
- [ ] `lopezdeprado2018` is cited in 3.6
- [ ] Table 1's expectations match the echo column in §4
- [ ] No claim of return predictability anywhere (red line D)
- [ ] The shared denominator in `u` is stated explicitly in 3.1
