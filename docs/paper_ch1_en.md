# Section 1 — Introduction (English, for review and editing)

> **Generated from `mac_irl_icaif26.tex` — do not hand-edit the prose above the design notes.**
> Regenerate with `python scripts/tex_to_md_mirror.py --section 1 --out docs/paper_ch1_en.md` after changing the manuscript.

---

## Introduction

On the same day, in the same stock, and facing the same public information, investors act differently. Which conditions tilt each investor type toward buying or toward selling? This is a basic question for price formation, for market microstructure, and for investor-protection policy, since type-level flow is where behavioral tendencies reach the market. The problem we address is how to recover, from observed daily flow alone, a description of what each investor type responds to—and to recover it in a form that can be compared across types.

Existing evidence does not deliver that comparison. The stylized regularities of type-specific behavior—momentum versus contrarian trading [grinblatt_keloharju2001], the disposition effect [shefrin_statman,odean1998], institutional herding [lakonishok_shleifer_vishny]—have been established one phenomenon at a time, in separate reduced-form regressions on different samples and specifications. Asset demand systems [koijen_yogo2019] do estimate type-level preferences within a single framework, but from quarterly holdings rather than daily flow, and without behavioral state variables. As a result, even a simple comparison—does foreign flow load more strongly on momentum than retail flow does?—has not been answered under a shared specification.

We approach this problem through inverse reinforcement learning [ziebart2008]. Each investor type is treated as an agent whose observed daily net-buy intensity reveals a latent reward defined over interpretable market-state features. A concave reward makes the optimal action graded, so trading intensity is derived rather than assumed. The three types are estimated *under a shared specification*—the same feature set, the same model form, the same penalty, with only the parameters learned separately from each type's own data—so that any difference between types reflects what is learned rather than what was designed, and the estimated coefficients lie on a common scale. In this myopic formulation, the estimator coincides exactly with a Lasso (\S3). We state this rather than obscure it, and build interpretation and validation on top of it: the recovered rewards are checked for out-of-sample stability under combinatorial purged cross-validation [lopezdeprado2018], for agreement with independently established behavioral regularities, and for what they fail to identify.

Applied to 973 trading days of Samsung Electronics (005930) flow, the recovered rewards separate sharply across types: two types load with opposite signs on the same signals, while the third shows no identified preference. We report this null alongside the positive findings.

Our contributions are as follows.

- **A framework for comparing investor preferences on a common scale.** Estimating three investor types under a shared specification—the same features, the same model form, the same penalty—makes their recovered weights comparable *in magnitude, not only in sign*. - **A validation procedure for recovered rewards.** We specify and apply a procedure that subjects the recovered weights to out-of-sample stability under resampling and re-estimation, to agreement with independently established behavioral regularities, and to explicit disclosure of what the model fails to identify. - **Empirical results on Korean investor flow.** Foreign and retail rewards oppose each other not only on medium-term momentum—consistent with existing evidence—but also in their response to market regimes, with sign consistency of 98--100% across 45 out-of-sample splits. Institutional flow yields no identified preference despite converged optimization.

The remainder of the paper is organized as follows. Section 2 reviews related work; Section 3 presents the model together with the estimation and validation methodology; Section 4 reports experiments and results; and Section 5 discusses implications, limitations, and future work, including the extension to structural reward recovery.
