# Section 3 — Method (English, for review and editing)

> **Generated from `mac_irl_icaif26.tex` — do not hand-edit the prose above the design notes.**
> Regenerate with `python scripts/tex_to_md_mirror.py --section 3 --out docs/paper_ch3_en.md` after changing the manuscript.

---

## Method

**[Figure]** Shared-specification preference recovery. The three investor types receive the same three state features, the same two market contexts, the same model form, and the same penalty; only $\theta_i$ is fitted separately, from each type's own flow. Yesterday's action re-enters as the persistence feature (dashed). The five validation procedures examine different sources of instability, and Section 4 attributes each claim to the procedure that supports it; fitted values are reported there rather than here.

### Setup

Let $\mathcal{I} = \{\textrm{foreign}, \textrm{institutional}, \textrm{retail}\}$ index investor types. For type $i$ on day $t$, write $V^{\mathrm{buy}}_{i,t}$ and $V^{\mathrm{sell}}_{i,t}$ for gross purchase and sale value, and define the normalized net-buy action

$$u_{i,t} = \frac{V^{\mathrm{buy}}_{i,t} - V^{\mathrm{sell}}_{i,t}} {V^{\mathrm{buy}}_{i,t} + V^{\mathrm{sell}}_{i,t}} \in [-1, 1].$$

Scaling net buying by the type's own gross traded value retains both direction and intensity, and places every type's action on the same $[-1,1]$ scale regardless of that type's trading size—the scale on which the policy in (2) is clipped. Each type is treated as an agent acting independently: the three are estimated separately rather than as players in a game, so no equilibrium restriction is imposed and none is recovered. The prediction target is $a_{i,t+1} = u_{i,t+1}$, and the state $s_{i,t} = (x_{i,t}, C_t)$ contains only information observable through day $t$. The three types share the same feature set $x_{i,t} \in \mathbb{R}^3$, the same market context $C_t \in \mathbb{R}^2$, and the same model form, with only the parameters fitted separately from each type's own data. We refer to this design as shared-specification preference recovery (SPR).

We estimate on Samsung Electronics (005930) over the five calendar years 2021--2025. The exchange-rate context standardizes the log rate against its preceding 252 trading days, so the 1{,}225 trading days in that window yield 973 state days once the first 252 are consumed as warm-up, running from 6 January 2022 to 29 December 2025. Fixing a single stock removes cross-stock heterogeneity from the comparison across types, at the cost of cross-sectional generalization.

### Reward and optimal action

Each type is modelled as choosing a daily action that maximizes a state-dependent reward. Writing $g_{i,t}$ for the reward gradient at state $s_{i,t}$,

$$R_i(a \mid s_{i,t}) = g_{i,t} a - \tfrac{1}{2}\kappa a^2, \qquad a^{*}_{i,t} = \operatorname{clip}(g_{i,t}, -1, 1),$$

with $\kappa$ normalized to one. Concavity is what makes intensity informative. Under a reward linear in $a$ the optimal action would sit at a bound every day, so observed magnitudes would carry no information about the state. With a concave reward the optimal action is graded, and *how much* a type trades is derived from the model rather than imposed by a link function chosen for convenience. This is what allows the fitted coefficients to be read as magnitudes rather than signs alone.

A reward linear in the state features implies a linear score, which is why the specification below takes the form it does rather than being one functional choice among many, and it places the present myopic model on a path toward structural recovery under a genuine planning horizon, which we return to in Section 5. It does not buy counterfactual transfer: nothing here licenses predicting behaviour under a market environment other than the one observed, and we make no such claim.

### Parameterization and reduction

The score separates an average feature sensitivity, a context-dependent adjustment of that sensitivity, and a direct context effect:

$$q_{i,t} = (\beta_i + B_i C_t)^{\top} x_{i,t} + \alpha_i^{\top} C_t .$$

Here $\beta_i \in \mathbb{R}^3$ is the sensitivity at mean context, $B_i \in \mathbb{R}^{3 \times 2}$ how that sensitivity shifts with context, and $\alpha_i \in \mathbb{R}^2$ the direct context effect when features are at their standardized means, so $\beta_i + B_i C_t$ is the effective feature weight on day $t$. No intercept is included. Writing $\operatorname{rvec}$ for row-major vectorization and $[ \cdot ;\cdot ]$ for vertical concatenation,

$$ w_{i,t} &= \bigl[ x_{i,t};\ \operatorname{rvec}(x_{i,t} C_t^{\top});\ C_t \bigr], \\ \theta_i &= \bigl[ \beta_i;\ \operatorname{rvec}(B_i);\ \alpha_i \bigr] \in \mathbb{R}^{11}, $$

so that $q_{i,t} = \theta_i^{\top} w_{i,t}$ and the predicted action is $\hat a_{i,t+1} = \operatorname{clip}(\hat\theta_i^{\top} w_{i,t}, -1, 1)$. Under the reading of Section 3, $q_{i,t}$ is the reward gradient $g_{i,t}$. The five coefficients in $\beta_i$ and $\alpha_i$ carry the interpretation; the six entries of $B_i$ are retained in every fit as conditional adjustments.

With a one-step horizon and a Gaussian policy centred on the optimal action, maximizing the log-likelihood of the observed actions is equivalent to minimizing squared error between the score and the realized action. Adding an $L_1$ penalty gives

$$\hat\theta_i = \arg\min_{\theta} \Bigl\{ \tfrac{1}{|\mathcal{D}_i|} \sum_{t \in \mathcal{D}_i} \bigl( w_{i,t}^{\top}\theta - a_{i,t+1} \bigr)^2 + \lambda_i \lVert \theta \rVert_1 \Bigr\},$$

where $\mathcal{D}_i$ is the training index set for type $i$. We state this reduction rather than obscure it: under a myopic horizon the estimator is a Lasso, and our contribution is the validation protocol of Section 3, not a new estimator. On our sample the reduction is exact rather than approximate. Clipping could in principle break the equivalence, but the saturation rate is zero for all three types in every split, so the constraint never binds and (5) coincides with the maximum-likelihood problem on the observed data.

Two limits on interpretation follow immediately, and we state them here rather than deferring them. Because the horizon is one step, the recovered reward is informationally equivalent to the conditional best response, and $\kappa$ is normalized to one, so magnitudes are expressed in units of that normalization and are comparable across types only because the same normalization is imposed on all three. And we read $\beta_i$ as a conditional association in aggregate flow, not as a structural preference, and do not separate preferences from beliefs or institutional constraints: a type that buys after positive returns may prefer trend exposure, may believe returns persist, or may operate under a mandate that produces the same flow.

### Features and sign expectations

Our three state features are specified in advance rather than selected by search; each operationalizes one of the regularities reviewed in Section 2.

**Recent price movement.** With $P_t$ the closing price, $x^{\mathrm{mom}}_t = \log(P_t / P_{t-20})$. Medium-term return persistence motivates the momentum feature [jegadeesh_titman]. A positive coefficient is consistent with trend-following and a negative one with contrarian trading, so we expect a positive sign for foreign flow and a negative sign for retail flow, with no strong prior for institutional flow.

**Flow persistence.** $x^{\mathrm{per}}_{i,t} = a_{i,t-1}$, the previous day's action on the same own-denominator definition as (1). The coefficient is therefore a first-order autoregression on behaviour. The grounding is the own-following component of Sias's decomposition of institutional demand [sias2004], which separates institutions following each other from institutions following their own lagged trades; at the level of type aggregates those two components cannot be separated, so we use only the measurable own-following component and do not describe this feature as herding. We expect a positive coefficient for all three types, following the long-range dependence Oh [oh2025] reports for the netted daily flow of each type in this market, and leave the ranking across types open, since that paper finds the cross-type ordering does not survive netting.

**Aggregate loss region.** This feature approximates whether held inventory was acquired above the current trade price. Let $Q^b$ and $Q^s$ be gross purchase and sale quantity, $M^b$ and $M^s$ the corresponding values, and $P^b = M^b / Q^b$ the purchase VWAP; type and date subscripts are suppressed and $[v]^{+} = \max(v, 0)$. With $\tilde H = \rho H_{t-1}$,

$$ O &= [\tilde H - Q^s]^{+}, & H &= [\tilde H + Q^b - Q^s]^{+}, & N &= H - O, \\ \bar P &= \frac{O \bar P_{t-1} + N P^b}{H}, & P^{\mathrm{vwap}} &= \frac{M^b + M^s}{Q^b + Q^s}, $$

where $\tilde H$ is depreciated inventory carried into day $t$, $O$ the prior inventory surviving sales, $N$ newly retained inventory, and $H = O + N$ total effective inventory. The feature is $x^{\mathrm{uw}}_{i,t} = \bigl[(\bar P_{i,t} - P^{\mathrm{vwap}}_{i,t}) / \bar P_{i,t}\bigr]^{+}$, set to zero when effective inventory or trading is absent, with $H_{i,0} = 0$. The forgetting factor $\rho = 0.98$ weights recent trades more heavily, a half-life of roughly 34 trading days. The disposition effect [shefrin_statman,odean1998] motivates this feature, defined relative to an estimated average cost. Because that average cost is an aggregate proxy built from gross trades rather than per-account realized gains, its sign can be compared with the disposition prediction but the effect is not directly identified. We expect a positive coefficient for retail flow, with no clear prior for the other two types.

**Market context.** Let $r_t$ be the one-day KOSPI 200 return and $e_t$ the won-dollar closing rate. With $\mu_{t-1,252}$ and $\sigma_{t-1,252}$ the mean and standard deviation of $\{\log e_s : s = t-252, \dots, t-1\}$, define $z^{\mathrm{FX}}_t = (\log e_t - \mu_{t-1,252}) / \sigma_{t-1,252}$ and $C_t = (r_t, z^{\mathrm{FX}}_t)^{\top}$. Context enters both through $B_i$, as an interaction with the features, and through $\alpha_i$, as a direct effect. Within each split, features and contexts are standardized on the training indices only and that transformation is applied to the test indices.

The expectations above are fixed before estimation, so Section 4 reads the fitted coefficients against them rather than rationalizing them afterwards.

### Estimation and uniqueness

Problem (5) is convex and coordinate descent attains a global solution. We verify uniqueness rather than assume it. With $n_i = |\mathcal{D}_i|$, $W_i = [w_{i,t}^{\top}]_{t \in \mathcal{D}_i}$ and $y_i = [a_{i,t+1}]_{t \in \mathcal{D}_i}$: when $\lambda_i = 0$ we check $\operatorname{rank}(W_i) = 11$; when $\lambda_i > 0$ we form $g_i = 2 W_i^{\top}(y_i - W_i \hat\theta_i) / n_i$ and the equicorrelation set $E_i = \{ j : |g_{i,j}| = \lambda_i \}$ and check $\operatorname{rank}(W_{i,E_i}) = |E_i|$; full column rank of $W_i$ implies the latter for every subset. Across all 45 splits and all three types the design matrix had full rank 11, with worst-case condition number 6.94.

The penalty is $\lambda_i = 0.005$ for every type. A common penalty is what makes coefficient magnitudes comparable across types: under type-specific penalties a smaller coefficient could reflect heavier shrinkage rather than weaker sensitivity, and the institutional null reported in Section 4 could not be distinguished from that artifact.

Fitting uses Adam [kingma2015] with seed 42 and 75 epochs for every type. Batch size and learning rate are set per type—256 at $10^{-3}$ for foreign, 2048 at $5\times10^{-4}$ for institutional, and 512 at $10^{-3}$ for retail—because the three action series differ in dispersion, with standard deviations of 0.257, 0.125, and 0.334 respectively. Since the objective is convex with a unique solution, these settings affect the path to the optimum but not the optimum itself, provided each fit reaches it; we therefore verify convergence from the per-split loss trajectories and coefficient increments rather than assume it. We also tested the alternative of forcing common batch size and learning rate on all three types: it destabilizes the institutional optimization, with the loss rising within some splits, and lowers out-of-sample correlation for two of the three types.

### Validation protocol

A single fitted coefficient vector is not interpreted on its own. Four procedures and a feature ablation examine different sources of instability, and Section 4 attributes each claim to the procedure that supports it.

**(V1) Combinatorial purged cross-validation.** We partition the sample into ten chronological folds and take every choice of two folds as the test set, giving $\binom{10}{2} = 45$ splits [lopezdeprado2018], with a one-day purge at test boundaries and a five-day embargo after each test block. Across splits the training set holds 765 to 775 days, the test set 194 to 197, and 1 to 14 days are excluded by purging and embargo. This is the primary protocol: it supplies the sampling distribution of $\beta_i$ and $\alpha_i$, the dispersion of out-of-sample performance, and the common basis on which the ablation and penalty comparisons below are run. Two test folds need not be adjacent, so some splits train partly on data later than their test block. That is the design intent of the procedure, which is to extract many partitions from a short series rather than to simulate live trading; accordingly V1 performance is the out-of-sample magnitude of a conditional association and not a forecast, and we do not interpret sign consistency as a $p$-value.

**(V2) Calendar-month block bootstrap.** Because the 45 splits share observations, sign consistency across them understates uncertainty. We therefore resample calendar months with replacement 200 times, refitting each time, and report 95% intervals; monthly blocks preserve within-month dependence in daily flow. This procedure fits in sample and produces no held-out predictions, so we use it only for coefficient intervals and never quote performance from it.

**(V3) Expanding walk-forward evaluation.** Since V1 admits splits that train on later data, we also impose strict chronology, training on all prior calendar years and testing on 2023, 2024, and 2025 in turn, with training sets of 242, 487, and 731 days. Scalers are fitted within each window's training range. This procedure answers two questions only: whether coefficient signs survive a strict temporal ordering, and whether performance is regime dependent. With three windows it supports no significance test, and because only the full specification is refitted it says nothing about the contribution of any individual feature.

**(V4) Penalty substitution.** To check dependence on the estimator we refit under an $L_2$ penalty over a grid of fifteen strengths, select one strength per type, and compare coefficient signs with the $L_1$ fits.

**Feature ablation.** We refit the model with each feature and each context removed in turn, on the same 45 splits, and compare pooled out-of-sample performance against the full specification using a paired block bootstrap over dates with block length 20 and 10{,}000 resamples, controlling the false discovery rate across variants and metrics. Pooling predictions across splits is required for the paired comparison and gives figures that are not directly comparable to the per-split averages reported for V1; the two are distinguished where both appear. We report three single-feature and two context removals.

Performance is summarized by direction accuracy with $\operatorname{sign}(0) = 0$, correlation between predicted and realized actions, MAE, RMSE, and out-of-sample $R^2$ against each split's training-mean action. Because action dispersion differs across types, we do not rank types by absolute MAE. Finally, we fix in advance what would count as failure. A coefficient is reported as identified only when its sign is consistent across the 45 splits of V1 *and* its V2 interval excludes zero; otherwise it is reported as unidentified rather than interpreted. V3 is reported alongside as evidence on temporal stability rather than on identification: with only three windows the procedure is weak in both directions—in our results one coefficient survives V3 while failing both V1 and V2. A feature whose removal does not degrade performance is reported as contributing nothing. Two earlier claims were withdrawn on these grounds.

---

## Design notes (remove before submission)

**Where each part came from**

| Subsection | Source | Nature of work |
| --- | --- | --- |
| 3.1 | 유민 draft 1.1 | Rewritten in English + the denominator paragraph is new and mandatory |
| 3.2 | — | **Entirely new.** The IRL layer; no draft source |
| 3.3 | draft 1.3 for (3)–(4) | Reduction argument and identification scope are **new** |
| 3.4 | draft 1.2 | Momentum and loss region transcribed; **flow persistence is entirely new** |
| 3.5 | draft 1.4 first half | Uniqueness verification reused; λ corrected |
| 3.6 | draft 1.4 second half | V1–V4 labels kept, every number corrected; ablation is new |

**Corrections applied against the draft**

| Draft said | Corrected to | Why |
| --- | --- | --- |
| $x^{cross}$ = other types' lagged flow | flow persistence $a_{i,t-1}$ | herd spec discarded (A-H1) |
| "lagging the action excludes the mechanical offset" | **deleted** | False — the common denominator makes the lagged cross feature ≈ −½ own lag ($r = -0.99$) |
| λ = 0 / 0.01 / 0.0003 per type | **λ = 0.005 unified** | Unified penalty is what makes magnitudes comparable |
| bootstrap 1,000 resamples | **200** | Actual run |
| walk-forward 10 expanding windows | **3 calendar years** | Actual run; do not claim 10 |
| "Symmetric Preference Recovery" | **Shared-specification** | "Symmetric" reads as symmetric distribution/matrix |
| no ablation | ablation section added | It is where the persistence feature's contribution is established |

**Draft assets deliberately preserved**

Lasso uniqueness verification (rank checks, worst condition number 6.86) · the loss-region recursion with $\rho = 0.98$ · the V1–V4 labelling · the hedge that sign consistency is not a $p$-value · the caution against ranking types by absolute MAE · the aggregate-proxy caveat on average cost.

**Open items — §4 handoff (updated 2026-07-30)**

Canonical is now `runs/continuous_reward3_persist_epochs75{,_validation}`: 75 epochs for all
three types, per-type batch and learning rate retained. Everything below is measured on it.

**Identification tiers under the §3.6 criterion (V1 sign consistency + V2 interval excluding zero).**
V3 is reported alongside but does not gate.

| Tier | Coefficients |
| --- | --- |
| Identified, temporally stable (no V3 reversal) | foreign momentum, foreign persistence, retail momentum, retail persistence |
| Identified, regime dependent (V3 reverses) | retail loss region, institutional persistence |
| Not identified | foreign loss region, institutional momentum, institutional loss region |

**Context main effects now have intervals** (α bootstrap added 2026-07-30). Five of six exclude zero,
and all four foreign/retail entries do — so contribution ③'s claim that the two types oppose each
other *also in their response to market regimes* is fully supported:

| | KOSPI 1-day return | FX level (252-day z) |
| --- | --- | --- |
| Foreign | **+0.0325** [+0.0178, +0.0483] ✔ | **−0.0225** [−0.0479, −0.0002] ✔ |
| Retail | **−0.0217** [−0.0424, −0.0059] ✔ | **+0.0322** [+0.0017, +0.0643] ✔ |
| Institutional | −0.0061 [−0.0119, −0.0002] ✔ | −0.0038 [−0.0122, +0.0003] ✘ |

- 🔴 **Report the institutional result as a null, and do not let the two marginal coefficients
  suggest otherwise.** Institutional persistence (+0.0099) and the KOSPI main effect (−0.0061)
  technically exclude zero, but their interval endpoints sit **0.00007 and 0.00016** from zero, and
  out-of-sample $R^2$ is **−0.003** — the model does worse than predicting the training mean.
  Direction accuracy is 0.539. Report them as boundary cases in a footnote, not as findings. The
  abstract's "no identified preference" stands as written; do **not** add "beyond flow persistence".

- 🔴 **§4 must carry the market-clearing hedge.** Nothing in §3 forecloses the reading "foreign and
  retail oppose each other, therefore we found strategic interaction." One type has to be on the
  other side of another's trades, so opposite signs can arise from accounting rather than behaviour.
  Supporting measurement: on the common scale of net purchases over the stock's *total* traded value
  the three types very nearly offset — mean $|\sum_j u_j| = 0.0133$ against mean $|u_{\text{foreign}}|
  = 0.1254$. Source: `experiments/2026-07-27/1932_herd_자기시차_공선성_진단.md`.

- **Ablation outcome changed for the better.** Under the previous under-trained fits, removing the
  loss region *improved* retail direction accuracy — an awkward result we would have had to explain.
  At convergence, removing it **significantly degrades** retail correlation ($q = 0.0322$).
  `remove_persist` remains the strongest single result (foreign, RMSE $q = 0.0007$, correlation
  $q = 0.0014$). Removing institutional momentum still yields a significant *improvement*
  (correlation $q = 0.0122$, RMSE $q = 0.0231$), supporting "institutional flow does not respond to
  momentum". Exclude the two degenerate group variants from the table.

- **Out-of-sample $R^2$** against each split's training-mean action: foreign **+0.130**,
  institutional **−0.003**, retail **+0.107**. Reproduce with
  `scripts/verify_paper_numbers.py --run-dir runs/continuous_reward3_persist_epochs75`.

- **$B_i$ stays at pattern level.** Most interaction coefficients fail to exclude zero; the only
  exceptions are foreign momentum × KOSPI and institutional loss region × KOSPI. §3.3 already commits
  to treating $B$ as adjustments rather than findings — honour that in §4 and list it in §5.

- **Convergence evidence is available** for the §3.5 claim: the last epoch contributes at most 0.4%
  of institutional and 0.1% of retail total loss reduction, and the largest final-epoch coefficient
  change is 1.7% and 1.2% of scale respectively. Source:
  `experiments/2026-07-30/verify_paper_numbers_epochs75/convergence_summary.csv`.

- Robustness grid ($\rho \times$ momentum window, 9 combinations) exists only for the **herd** spec.
  Either re-run under the current canonical or drop it from §4.

- §2's Oh (2025) sentence reports persistence strongest for **retail**; our AR(1) says **foreign**
  (+0.401 vs +0.345). Soften the ordering or state the difference in statistic, sample, and stock.

- §3.1's sample sentence was rewritten to present 2022–2025 as a period choice. **The actual reason
  for that window is still unconfirmed** — ask 유민 whether it was inherited. The KOSPI 200 series is
  not the constraint; it has no missing values.








