# Section 2 — Related Work (English, for review and editing)

> **Generated from `mac_irl_icaif26.tex` — do not hand-edit the prose above the design notes.**
> Regenerate with `python scripts/tex_to_md_mirror.py --section 2 --out docs/paper_ch2_en.md` after changing the manuscript.

---

## Related Work

To place our contribution, we review four lines of work and note what each leaves open.

**Empirical evidence on investor-type flows.** That trading behavior differs systematically across investor types is well documented across markets [choe_kho_stulz,grinblatt_keloharju2001,kaniel_saar_titman]. Korean foreign flow shows positive-feedback trading and daily herding [choe_kho_stulz], and short-horizon contrarian patterns appear in US individual flow [kaniel_saar_titman]. The sharpest evidence comes from a Finnish register recording every investor's daily buys, sells, and holds: Grinblatt and Keloharju [grinblatt_keloharju2001] find contrarian behavior strongest among households, while foreign investors tend to be momentum traders. The topic remains active—Oh [oh2025] applies detrended fluctuation analysis to Korean daily flows from 2015 to 2024 and finds long-range dependence in all three types, with the cross-type ranking visible in gross buys and sells attenuating once flows are netted. These findings, however, either characterize the time-series properties of flow or come from separate regressions on different samples and specifications, each targeting one phenomenon at a time, so the resulting coefficients cannot be placed side by side across types.

**Behavioral regularities behind type differences.** Three regularities recur across this literature. Medium-term return persistence underlies momentum and contrarian trading [jegadeesh_titman]. The disposition effect—reluctance to realize losses [shefrin_statman,odean1998]—is documented in US brokerage accounts and recovered from daily Finnish trades [grinblatt_keloharju2001]. Herding measures [lakonishok_shleifer_vishny] capture participants *of the same type* trading a stock simultaneously; Sias [sias2004] separates institutions that follow each other from institutions that follow their own lagged trades. Each regularity has been established in isolation, for one phenomenon and often one investor type, so none of them indicates how strongly a given type weighs one regularity against another.

**Demand systems for heterogeneous investors.** A separate line estimates type-level preferences within a single framework. Koijen and Yogo [koijen_yogo2019] develop a characteristics-based demand system and estimate it on Form 13F holdings, recovering how demand varies across institution types and what that heterogeneity implies for prices. The estimation applies a shared specification to all investors, as we require, but it operates on *quarterly* holdings and on firm fundamentals rather than the behavioral state variables where the regularities above are defined. Daily flow falls outside its scope.

**Inverse reinforcement learning and preference estimation in finance.** Inverse reinforcement learning recovers a reward that rationalizes observed behavior, with non-uniqueness resolved by the maximum-entropy principle [ziebart2008]. In finance the dominant use is learning or improving trading strategies: a sentiment-based reward learned with Gaussian process IRL to drive a trading system [yang2018gpirl], strategy recovery under transaction costs [sun2023transaction], and—closest to our setting—Halperin et al. [halperin2022], who recover the implied reward of individual fund managers and feed it to a forward RL algorithm to improve their allocations. Outside finance the same machinery has been used to interpret behavior rather than to trade, for instance to characterize risk-prone and risk-averse decision making [liu2019risk]. The shared objective in this line is a profitable policy or the reward of a single actor, recovered to be imitated or improved. To our knowledge, no prior work estimates the rewards of *heterogeneous investor types* on a shared feature set so that they can be compared with one another, nor subjects the recovered rewards themselves to out-of-sample stability and construct-validity testing.

---

## Design notes (remove before submission)

**How each block ends** — every block closes with what the prior line leaves open:

| Block | Closing gap | Maps to (contribution) |
| --- | --- | --- |
| Investor-type flows | separate regressions / time-series characterization → coefficients not comparable | comparability, empirics |
| Demand systems | quarterly holdings, no behavioral state variables | comparability |
| IRL in finance | single actor, recovered to imitate or improve; no cross-type comparison, no validation | comparability, validation |

*(The "Behavioral regularities" row was removed with the block — it is now §3's concern.)*

**Second-tier trims, if §4 needs more room**

| Where | Current | Candidate | Saves |
| --- | --- | --- | --- |
| Block 2, Koijen–Yogo | "characteristics such as market equity, book equity, profitability, investment, dividends, and market beta" | "firm fundamentals rather than behavioral state variables" | ~8 words |
| Block 1, Oh 2025 | method + data + period all spelled out | keep one, drop two | ~10 words |
| Block 2 overall | 86 words | **do not touch — already the leanest block** | — |

**Sign expectations carried into §4** (Table 1 lives there)

| Feature | Foreign | Retail | Institution | Basis |
| --- | --- | --- | --- | --- |
| Momentum (20d return) | **+** trend-following | **−** contrarian | weak — little positive-feedback evidence | GK2001, CKS1999, KST2008, LSV1992 |
| Loss region (disposition) | unclear | **+** buys/holds below average cost | weak — literature centers on individuals | Odean1998, Shefrin–Statman1985, GK2001 |
| Herding (other types' prior flow) | no directional expectation | no directional expectation | no directional expectation | literature measures *within*-type herding (LSV1992, Sias2004); cross-type sign can be mechanical under market clearing (GK2001) |

**References newly added on 2026-07-27**

| Key | Source | Placed in | Why |
| --- | --- | --- | --- |
| `oh2025` | Oh, "Nonlinear Evidence of Investor Heterogeneity," arXiv:2508.20426 | block 1 | Korean type-segregated daily flow, 2015–2024. Shows the topic is current; measures *persistence*, not preferences, so our gap holds |
| `sias2004` | Sias, "Institutional Herding," RFS 17(1), 165–206 | block 3 | Separates institutions following *each other* from following their *own* lagged trades — grounds our within-type vs cross-type distinction |
| `halperin2022` | Halperin, Liu, Zhang, arXiv:2201.01874 | block 4 | Closest prior work: IRL recovers *fund managers'* implied reward. Cited explicitly so the difference (single actor, improve-not-compare) is on the record |
| `sun2023transaction` | Sun, Gong, Si, *Applied Intelligence* 53, 28186–28206 | block 4 | Author names confirmed from the PDF; resolves the open item |
| `liu2019risk` | Liu, Wu, Liu, ICML RWSDM Workshop 2019 | block 4 | IRL used for behavioral *interpretation* rather than strategy learning — same spirit as ours, outside finance |

**Reviewed and NOT used**

| Source | Reason |
| --- | --- |
| `momentum.pdf` | Identical to Choe–Kho–Stulz (already cited) |
| `Towards IRL for Limit Order Book Dynamics` (×2) | Same file uploaded twice (identical MD5); would be an optional extra citation at best |
| DeepTrader (Wang et al.) | Forward deep RL for portfolio management — neither IRL nor preference estimation; blurs the focus of block 4 |
| Lin, Beling, Cogill (multi-agent IRL, zero-sum games) | Game-theoretic MIRL; we estimate the three types independently, not as a game |
| Maeda et al. (latent segmentation) | Usable, but block 4 already carries five citations |

**Verified source claims** (PDF-checked)

- **CKS 1999**: positive-feedback trading by foreigners *before* the crisis; herding >20% for large *past winners* (daily) vs <5% for US mutual funds (quarterly, Wermers 1998). Conclusion is **no evidence of destabilization** — do not cite it the other way. The 20%/5% figures are recorded here for reference only; they are **no longer used in the text** (see Open items). Sample is 1997, ~25 years before ours — `oh2025` carries the "still true today" load.
- **GK2001**: "Contrarian behavior is greatest for the household… Foreign investors, by contrast, tend to be momentum investors." Also "not all investors can be contrarians if all buys are sells and vice versa" — our herd caveat.
- **LSV 1992**: 769 tax-exempt (mostly pension) funds; "pension managers **do not strongly pursue** these potentially destabilizing practices." Cite as a *weak* institutional expectation, not as evidence of herding.
- **Sias 2004**: institutional demand this quarter correlates with last quarter; decomposed into following each other vs following own lagged trades; herding "differs across capitalizations and investor types."
- **Koijen–Yogo 2019**: 13F holdings are **quarterly**, managers >$100M since 1980; characteristics are market equity, book equity, profitability, investment, dividends, market beta.
- **Oh 2025**: DFA on Korean buy/sell/net flows 2015–2024; persistence strongest for retail, weakest for foreign.
- **Shefrin–Statman 1985**: *JF* 40(3), 777–790 — bibliographic details confirmed from the JSTOR scan.

**Open items**

- ~~Consider naming Wermers (1998) explicitly for the "quarterly estimates for US mutual funds" comparison.~~ **Resolved 2026-07-27 — comparison clause cut.** It rested on a 1998 quarterly benchmark, overstated the ratio ("an order of magnitude" vs. the actual ~4×), said "large stocks" where CKS says large *past winners*, and established a fact (foreign herding is strong) that we never use. Cutting it also removes a slight tension with block 3, where we distance our cross-type feature from the literature's within-type herding measures.
- `kingma2015` (Adam) is in the bibliography but not yet cited — it belongs in §3.4.





