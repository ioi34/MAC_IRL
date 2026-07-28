# Paper checklist — ICAIF 2026

> Target: **ICAIF 2026**, submission deadline **2026-08-02 AOE**. Milan, Nov 14–17.
> Source of truth for the manuscript: `acmart-primary/mac_irl_icaif26.tex` (English).
> Working convention: edit the English `.tex` only; propagate to other files when Tony says 확정.
> Korean version (`mac_irl_icaif26_kr.tex`) — **do not touch.**
>
> **Note on this file (2026-07-27):** the previous checklist was not present on disk when
> this version was written, so items below were re-entered from the current state of the
> repo and the manuscript. Items marked *(재구성)* were reconstructed from memory of prior
> decisions and should be confirmed against the `.tex` before being trusted. Item A-H1 is
> newly measured and fully sourced.

---

## A. Experiments

### A-H1 🔴 BLOCKING — herd feature is not identified as a cross-type quantity

> 📄 **Full working brief: `docs/herd_identification_brief.md`** — self-contained, written so a
> fresh session can pick this up cold. Includes the cause, the evidence, the already-available
> `remove_herd` ablation comparison, the options with costs, concrete run commands, and the
> list of paper sentences frozen pending this decision.

**Status:** open, discovered 2026-07-27. Blocks §2 block 3, §4 herd reporting, and the
sign-expectation table.

**⭐ The headline is already known to be safe.** The validation suite already contains a
`remove_herd` ablation. Dropping herd leaves foreign momentum at +0.0639 (was +0.0476) and
retail at −0.0325 (was −0.0301), both 100% sign-consistent, and retail's disposition weight
holds at 100% while foreign's weakens further (89% → 76%). The opposing-momentum result and
the retail-only disposition contrast do not depend on the herd specification. No rerun was
needed to establish this — see the brief, §3.

**What was found.** `src/data/preprocess.py:53` defines

```python
u_{investor} = net_buy_{investor} / trading_value
```

with `trading_value` the **stock's total** trading value — a denominator shared by all three
types. The three `u` series therefore nearly sum to zero (mean |Σu| = **0.0133** vs
mean |u_foreign| = **0.1254**, i.e. residual ≈ 10% of scale). Since `src/features/herd.py:9`
builds `herd_i(t)` as the mean of the other two types' `u(t−1)`, it follows that
`herd_i(t) ≈ −½·u_i(t−1)`.

Measured collinearity with the type's **own** lagged flow:

| 대상 | 피처/가중치 | 값 | 변동성/일관성 | 출처 |
| --- | --- | ---: | ---: | --- |
| 외국인 | corr(herd(t), own u(t−1)) | −0.989 | — | 진단 스크립트 (raw CSV) |
| 기관 | corr(herd(t), own u(t−1)) | −0.974 | — | 진단 스크립트 (raw CSV) |
| 개인 | corr(herd(t), own u(t−1)) | −0.992 | — | 진단 스크립트 (raw CSV) |

**Why the recovered herd coefficients look mechanical.** Own flow is positively
autocorrelated, so a negative herd coefficient is what an accounting identity alone would
produce, with magnitude ordered by AR(1):

| 대상 | 피처/가중치 | 값 | 변동성/일관성 | 출처 |
| --- | --- | ---: | ---: | --- |
| 외국인 | AR(1) of own u | +0.394 | β_herd = −0.0382 | `runs/continuous_reward3_lambda_unified/` |
| 개인 | AR(1) of own u | +0.365 | β_herd = −0.0319 | `runs/continuous_reward3_lambda_unified/` |
| 기관 | AR(1) of own u | +0.169 | β_herd = −0.0047 | `runs/continuous_reward3_lambda_unified/` |

Sign is negative for all three and the magnitude ordering matches the AR(1) ordering exactly.
The 100% sign consistency is therefore **not** evidence of a behavioral regularity.

**Momentum contamination (headline appears to survive).** corr(herd, mom20) = −0.435
(foreign) / +0.407 (retail) / −0.076 (institution). Independently, own lagged flow vs mom20 =
+0.425 (foreign) / −0.416 (retail), i.e. a second measurement pointing the same way as the
headline momentum result (foreign +0.0476 vs retail −0.0301).

**Caveat on the numbers above.** Computed from `data/raw/samsung_macirl_EXTENDED_2019_2025.csv`
with `u` and `mom20` reconstructed in the diagnostic script and a 2022-01-01 cutoff, giving
n = 977 (the paper reports 973). Correlation and autocorrelation facts are structural and will
not change, but **the exact figures must be recomputed from the canonical run's feature
matrix before they enter the paper.**

**To do**

- [ ] Recompute A-H1 diagnostics from the canonical run feature matrix (not the raw CSV), to fix n and the exact values
- [ ] Decide the specification (options below) — **not yet decided**
- [ ] Re-run the canonical run and the validation suite under the chosen specification
- [ ] Confirm momentum coefficients are stable across old and new specification (this is the check that protects the headline)
- [ ] Rewrite §2 block 3 motivation and §4 herd reporting to match
- [ ] Update the sign-expectation table

**Options considered (2026-07-27)**

| Option | Change | Cost | Note |
| --- | --- | --- | --- |
| A | Relabel only, no re-run — report as negated own-flow persistence | none | Feature is still literally other types' flow; a reviewer who computes r will ask why own lag wasn't used |
| **B (추천)** | Swap `herd` → `persist` (`src/features/persist.py`, already implemented) | config + re-run | Keeps 3 features, all interpretable, removes the clearing confound. ⚠️ Oh (2025) reports persistence strongest for **retail**; our AR(1) says **foreign** — DFA Hurst ≠ AR(1), different sample/stock, handle carefully |
| C | Use one other type only (`herd_a`/`herd_b`, already implemented) | config + re-run | corr(foreign, institution) = −0.048 → that pair *is* identified, but retail pairs are −0.818 / −0.522. Identification quality differs by type, which **breaks the symmetric-estimation selling point** |
| D | Drop to 2 features | re-run | Loses the control; persistence may leak into momentum |

---

### A-2 Bootstrap resamples *(재구성)*

- [ ] Re-run block bootstrap at 200 resamples (currently 100)

---

## B. Writing, by section

- [x] Abstract — drafted (~205 words), in `.tex`
- [x] §1 Introduction — drafted (~499 words); restructured after advisor feedback ("서론이 너무 산발적"); contributions now 3 bullets with no C1/C2/C3 labels
- [x] §2 Related Work — **3** bold run-in blocks (Pattern A), **455 words ≈ 0.48 page**. See B-§2-LEN below
- [ ] §3 Method — in progress, 권유민. **Plan and hand-off spec: `docs/paper_ch3_plan.md`.** Six subsections, ~1,530 words ≈ 1.9 pages. 3.1–3.3, 3.5, 3.6 are writable now; **3.4 is partially blocked by A-H1**. `kingma2015` (Adam) belongs in **3.5**, not 3.4. The paragraph moved from §2 sits in the `.tex` as raw material and must be dissolved, not pasted — see B-§2-LEN

### B-§2-LEN ✅ Resolved 2026-07-27 — §2 length and the misplaced block

**Page budget.** ICAIF 2026 allows **8 pages total in two-column sigconf, including all
figures and references, no exceptions**
([CFP](https://icaif2026.org/call-for-papers.html)). Rendering measures ≈ 950 words/page.

**Measured before the move** (644 words ≈ 0.68 page):

| Block | Words |
| --- | ---: |
| lead-in | 19 |
| ① Empirical evidence on investor-type flows | 156 |
| ② Demand systems for heterogeneous investors | 86 |
| ③ Behavioral regularities behind our features | **189** |
| ④ IRL and preference estimation in finance | 171 |

**Diagnosis.** 0.68 page is within the normal 0.5–0.75 range for an 8-page paper, so §2 was
not abnormally long in absolute terms. The real problems were (a) §4 needs the room — the
validation suite carries 3–4 tables — and (b) **block ③ was misplaced.** Blocks ①②④ survey
prior work; ③ justifies *our* feature choices, which is method motivation. It was also the
longest block. Tellingly, the lead-in promised "three lines of work" while four blocks were
present.

**Action taken.** Block ③ moved into `\section{Method}` with a hand-off comment for 유민 and
the A-H1 HOLD notice preserved. Result: §2 = **455 words ≈ 0.48 page**, three blocks, lead-in
now accurate, ~0.2 page returned to the budget. Nothing was discarded — §3 has to define the
features anyway.

**Watch:** §2 no longer cites `jegadeesh_titman`, `shefrin_statman`, `odean1998`,
`lakonishok_shleifer_vishny` (all moved to §3). Behavioral coverage in §2 now rests on block ①
(GK2001, CKS, KST, Oh), which is sufficient.

**Second-tier trims, only if §4 runs out of room:** Koijen–Yogo's six-characteristic list
(~8 words), Oh 2025's method+data+period triple (~10 words). Block ② is the leanest at
86 words — leave it alone.
- [ ] §4 Results
  - [ ] Put the sign-expectation column beside the recovered β
  - [ ] Report λ-unified numbers throughout
  - [ ] **Delete the old §4.7 price-impact section** (claim was refuted — see `experiments/2026-07-24/1745_동시점_가격영향_정렬오류_반증.md`)
  - [ ] Herd reporting blocked on A-H1
- [ ] §5 Conclusion — mention the extension path here rather than in §1 P3

---

## C. Format and infrastructure

- [ ] Convert `thebibliography` to BibTeX with `ACM-Reference-Format`
- [ ] **Remove the `review` option** from `\documentclass` before submission
- [ ] 🔴 **Delete stray build files in `acmart-primary/`: `_en_build.*`, `_kr_build.*`, `_try.*`, `preview_mac_irl_icaif26.pdf`.** Not cosmetic — **these actively hijack builds.** On 2026-07-27 a build with `TEXINPUTS=/…/acmart-primary//:` compiled the *stale* `acmart-primary/_en_build.tex` instead of the fresh local copy, producing a PDF with an old title ("Comparing Investor Preferences on a Common Scale") and an old disposition claim that violates red line D. The build looked completely successful. **Always put `.` first: `TEXINPUTS=".:/…/acmart-primary//:"`, and give the scratch file a unique name.**
- [ ] Remove the "Design notes (remove before submission)" section from the `docs/*.md` mirrors before they are used as copy
- [ ] Keep `\PassOptionsToPackage{expansion=false,protrusion=false}{microtype}` only in the throwaway build copy, not in the submission `.tex`

### C-git 🔴 Blocked

Branch is `tony`; `docs/` and `.claude/` are untracked and nothing in `docs/` is committed.
A stale `.git/index.lock` blocked earlier attempts and the delete permission was declined.

- [ ] Tony to run `rm -f .git/index.lock` locally
- [ ] Commit `docs/` and `experiments/2026-07-24/`, `2026-07-26/`, `2026-07-27/`
- [ ] Push to `main`

**Also missing from `docs/`:** `paper_ch1_en.md`, `paper_ch1_kr.md`, `refs_checklist.md` were
expected but are not on disk (only `paper_ch2_en.md` is present). Since nothing was ever
committed, they cannot be recovered from git. Decide whether to regenerate them from the
`.tex` — the `.tex` itself is intact, so no manuscript content is lost.

---

## D. Claim red lines — do not cross

- ❌ **No return-predictability claim.** The abstract says so explicitly; keep it that way.
- ❌ **No contemporaneous price-impact claim.** Tested and refuted; the earlier +0.30/−0.30 figures were never produced by any run.
- ❌ **No disposition-effect claim in the contributions.** Retail passes the block bootstrap but fails walk-forward; foreign is the reverse. Neither is robust enough to headline.
- ❌ **Do not cite CKS (1999) as evidence of destabilization** — its conclusion is the opposite.
- ❌ **Do not cite LSV (1992) as evidence of institutional herding** — it finds pension managers "do not strongly pursue" such practices. It is a *weak* institutional expectation.
- ❌ **No behavioral interpretation of the herd coefficient** until A-H1 is resolved.
- ⚠️ Under double-blind, "we" is fine, but self-citation must be third person.

---

## E. Locked decisions

| Decision | Value |
| --- | --- |
| Venue / deadline | ICAIF 2026 / 2026-08-02 AOE |
| Title | *Same Signals, Different Rewards: Recovering and Validating Investor-Type Preferences from Daily Flow via Inverse Reinforcement Learning* |
| Model | **Continuous** (CLAUDE.md rule 6 — default for all experiments) |
| Feature count | 3 reward features + 2 contexts *(third feature under review — A-H1)* |
| λ | 0.005, **unified across all three types** (legitimizes "comparable in magnitude") |
| CPCV | 10 folds / 2 test / purge 1 / embargo 5 = 45 splits; train-only standardization; seed 42 |
| Sample | Samsung Electronics (005930), 973 trading days |
| Canonical run | `runs/continuous_reward3_lambda_unified/` |
| Validation run | `runs/continuous_reward3_lambda_validation/` |
| §2 format | **Pattern A — bold run-in headings, no numbered subsections.** Re-examined 2026-07-27 against two ICAIF papers and confirmed. Numbered subsections earn their heading at ~300 words per unit: the ICAIF '25 MARL paper (`3768292.3770411`) runs 892 words over 3 subsections (347 / 349 / 137) in a 9-page paper, while ICAIF '26 `033.pdf` uses run-in headings at ~85 words per block. Ours are 156 / 94 / 86 / 171 (avg 127) in 549 words — run-in territory. Converting would require merging to 2 subsections, which blurs the demand-system gap that §1 names separately. §3 keeping numbered subsections (150–350 words each) while §2 uses run-in is form following unit size, and is standard. |
| §2 block order | ① type flows → ③ behavioral regularities → ② demand systems → ④ IRL. ③ must precede ②, because ②'s "the regularities above are defined" refers to it |
| Contributions | 3 bullets, no C1/C2/C3 labels, no gap restatements |
| §3-D1 | **Validation protocol specified in §3; §4 reports results only** — matches "we specify and apply a procedure" and contribution #2 |
| §3-D2 | **3.4 runs definition → grounding → sign expectation per feature**, plus a summary table — keeps each formula next to its justification |
| §3-D3 | **Sign expectations stated in §3, echoed in §4** — pre-commitment is what makes §4 a test rather than a rationalization |
