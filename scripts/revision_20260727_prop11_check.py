"""명제 1.1 재검토: 비볼록 clip 목적식 vs 제약 Lasso, 그리고 유일성.

심사 지적: 식 (6)의 예측값은 clip(Z theta) 이므로 목적식은 전역적으로 비볼록이다.
보고된 해에서 |q|<1 인 것은 그 해 근방에서만 Lasso 와 같음을 보일 뿐, 포화영역에
더 좋은 해가 없음을 보장하지 않는다. 또한 볼록성만으로 유일성이 따라오지 않는다.

본 스크립트는 네 가지를 계산한다 (모두 학습 인덱스 기준, 45분할 x 3유형).

1) 엄격 실현가능성 (strict feasibility)
   C = {theta : max_t |z_t' theta| <= 1} 위에서 정의한 제약 Lasso 를 추정량으로 삼을 때,
   무제약 Lasso 해가 C 의 내부에 있는지. 내부점이면 KKT 에서 제약 승수가 0 이므로
   두 문제의 해가 일치한다.

2) 유일성
   - lambda = 0 : rank(Z) = p 이면 목적식이 강볼록 -> 유일. sigma_min, cond 보고.
   - lambda > 0 : equicorrelation set E = {j : |(2/n) z_j'(y - Z theta)| = lambda} 에 대해
     rank(Z_E) = |E| 이면 Lasso 해가 유일 (Tibshirani 2013). 이를 직접 확인한다.

3) 포화영역에 대한 증명 가능한 경계
   임의의 theta 에 대해 S(theta) = {t : |z_t' theta| > 1} 라 하면
       F(theta) >= (1/n) sum_{t in S} (1 - |a_t|)^2
   이므로, F(theta) < F(theta_hat) 인 theta 는 S 의 크기가
       k* = max{ k : (1/n) * (k 개 최소 (1-|a_t|)^2 의 합) < F(theta_hat) }
   이하여야 한다. 완전 포화해의 목적값 하한은 mean((1-|a_t|)^2) 이다.
   (증명: 포화된 t 의 손실은 (sign - a_t)^2 >= (1-|a_t|)^2, 비포화 t 의 손실은 >= 0,
    L1 항은 >= 0.)

4) 다중시작 전역 탐색
   비볼록 목적식 F 를 근사적 근사경사(proximal gradient)로 여러 시작점에서 최소화한다.
   시작점: Lasso 해의 배율 확대(포화영역 진입), 무작위 방향 x 여러 노름,
   sign(a) 를 맞추는 선형분류 방향(완전 포화해의 최선 후보).
   목적: theta_hat 보다 나은 해가 발견되는지.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.revision_20260727_core import (
    LAMBDA_CFG,
    build_design,
    fit_exact,
    load_base,
    make_feature_tensor,
)
from src.data.splits import build_cpcv, combine_test_folds
from src.features.scaling import (
    fit_context_scaler,
    fit_feature_scaler,
    transform_context_matrix,
    transform_feature_tensor,
)

OUT = Path("runs/revision_20260727")
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260727)


def objective(Z, y, theta, lam):
    """F(theta) = (1/n)||clip(Z theta) - y||^2 + lambda ||theta||_1  (식 (6))."""
    pred = np.clip(Z @ theta, -1.0, 1.0)
    return float(np.mean((pred - y) ** 2) + lam * np.abs(theta).sum())


def objective_batch(Z, y, Theta, lam):
    """여러 시작점의 F 를 한 번에. Theta: (S, p) -> (S,)"""
    Q = np.clip(Theta @ Z.T, -1.0, 1.0)
    return np.mean((Q - y) ** 2, axis=1) + lam * np.abs(Theta).sum(axis=1)


def prox_grad_batch(Z, y, Theta0, lam, n_iter=300, step=None):
    """비볼록 F 에 대한 근사경사법을 모든 시작점에 대해 동시 수행(벡터화).

    포화 구간에서는 평활항의 기울기가 정확히 0 이므로 완전 포화 시작점은 그 자리에
    머문다. 이는 결함이 아니라 목적식의 구조(포화 고원)를 그대로 반영한 것이며,
    고원 위의 목적값은 시작점 평가로 이미 포착된다."""
    n = len(y)
    if step is None:
        step = 1.0 / (2.0 * np.linalg.norm(Z, 2) ** 2 / n + 1e-12)
    Theta = np.array(Theta0, dtype=float, copy=True)
    best_val = objective_batch(Z, y, Theta, lam)
    best = Theta.copy()
    for _ in range(n_iter):
        Q = Theta @ Z.T
        interior = (np.abs(Q) < 1.0)
        resid = np.clip(Q, -1.0, 1.0) - y
        grad = (2.0 / n) * (resid * interior) @ Z
        Theta = Theta - step * grad
        if lam > 0:
            Theta = np.sign(Theta) * np.maximum(np.abs(Theta) - step * lam, 0.0)
        val = objective_batch(Z, y, Theta, lam)
        improved = val < best_val
        if improved.any():
            best_val = np.where(improved, val, best_val)
            best[improved] = Theta[improved]
    return best, best_val


def saturation_bounds(y, f_hat):
    """포화 관측 수에 대한 증명 가능한 상한 k* 와 완전 포화해의 하한."""
    n = len(y)
    penalties = np.sort((1.0 - np.abs(y)) ** 2)      # 오름차순
    cum = np.cumsum(penalties) / n
    k_star = int(np.searchsorted(cum, f_hat, side="left"))  # cum[k-1] < f_hat 인 최대 k
    return {
        "k_star_max_saturated": k_star,
        "k_star_share_pct": 100.0 * k_star / n,
        "full_saturation_lower_bound": float(np.mean((1.0 - np.abs(y)) ** 2)),
        "max_abs_action": float(np.abs(y).max()),
    }


def uniqueness(Z, y, theta, lam, tol=1e-8):
    n, p = Z.shape
    s = np.linalg.svd(Z / np.sqrt(n), compute_uv=False)
    rank = int((s > tol * s[0]).sum())
    info = {
        "n_train": n, "p": p, "rank": rank, "full_rank": rank == p,
        "sigma_min": float(s[-1]), "cond": float(s[0] / s[-1]),
    }
    if lam > 0:
        resid = y - Z @ theta
        corr = np.abs((2.0 / n) * Z.T @ resid)
        E = np.flatnonzero(corr >= lam - 1e-7)          # equicorrelation set
        if len(E) == 0:
            info.update({"equicorr_size": 0, "equicorr_rank": 0, "lasso_unique": True})
        else:
            rE = int(np.linalg.matrix_rank(Z[:, E], tol=1e-8))
            info.update({"equicorr_size": int(len(E)), "equicorr_rank": rE,
                         "lasso_unique": rE == len(E)})
        info["support_size"] = int((np.abs(theta) > 1e-12).sum())
    else:
        info.update({"lasso_unique": rank == p, "support_size": int((np.abs(theta) > 1e-12).sum())})
    return info


def main() -> None:
    config, features, contexts, actions, dates, investors, fnames, cnames, lagged = load_base()
    cv = build_cpcv(config)
    split_input = np.arange(len(features)).reshape(-1, 1)

    rows, search_rows = [], []
    for split_id, (train_idx, test_folds) in enumerate(cv.split(split_input)):
        cs = fit_context_scaler(contexts, train_idx)
        sc = transform_context_matrix(contexts, cs).astype(np.float64)
        for inv_idx, inv in enumerate(investors):
            scaler = fit_feature_scaler(features[:, inv_idx], train_idx)
            scaled = transform_feature_tensor(features[:, inv_idx], scaler).astype(np.float64)
            Z = build_design(scaled, sc, 3)[train_idx]
            y = actions[train_idx, inv_idx]
            lam = LAMBDA_CFG[inv]

            theta = fit_exact(Z, y, lam)
            f_hat = objective(Z, y, theta, lam)
            q = Z @ theta
            row = {
                "split": split_id, "investor": inv, "lambda": lam,
                "F_hat": f_hat,
                "max_abs_q_train": float(np.abs(q).max()),
                "strictly_feasible": bool(np.abs(q).max() < 1.0),
                "n_saturated": int((np.abs(q) > 1.0).sum()),
                **uniqueness(Z, y, theta, lam, ),
                **saturation_bounds(y, f_hat),
            }
            rows.append(row)

            # ---- 다중시작 전역 탐색 (벡터화)
            starts, tags = [], []
            for c in (1.5, 2.0, 3.0, 5.0, 10.0, 50.0, 200.0):
                starts.append(c * theta); tags.append("scaled_lasso")
            theta_cls, *_ = np.linalg.lstsq(Z, np.sign(y), rcond=None)
            for c in (0.5, 1.0, 2.0, 5.0, 20.0, 100.0):
                starts.append(c * theta_cls); tags.append("classifier")
            for _ in range(300):
                d = RNG.normal(size=Z.shape[1]); d /= np.linalg.norm(d)
                starts.append((10 ** RNG.uniform(-1.0, 2.5)) * d); tags.append("random")
            Theta0 = np.vstack(starts)
            v_raw = objective_batch(Z, y, Theta0, lam)
            _, v_ref = prox_grad_batch(Z, y, Theta0, lam, n_iter=300)
            j_raw, j_ref = int(np.argmin(v_raw)), int(np.argmin(v_ref))
            best_val, best_tag = f_hat, "lasso"
            if v_raw[j_raw] < best_val:
                best_val, best_tag = float(v_raw[j_raw]), tags[j_raw] + "_raw"
            if v_ref[j_ref] < best_val:
                best_val, best_tag = float(v_ref[j_ref]), tags[j_ref] + "_refined"
            search_rows.append({
                "split": split_id, "investor": inv, "n_starts": len(starts),
                "F_hat": f_hat, "F_best_found": best_val, "best_source": best_tag,
                "improvement": f_hat - best_val,
                "beaten": bool(best_val < f_hat - 1e-10),
                "F_best_fully_saturated_start": float(v_raw[:7].min()),
            })
        if split_id % 15 == 0:
            print(f"[prop11] split {split_id} done", flush=True)

    df = pd.DataFrame(rows)
    sr = pd.DataFrame(search_rows)
    df.to_csv(OUT / "prop11_feasibility_uniqueness.csv", index=False)
    sr.to_csv(OUT / "prop11_global_search.csv", index=False)

    summary = {
        "splits": int(df.split.nunique()),
        "all_strictly_feasible": bool(df.strictly_feasible.all()),
        "max_abs_q_train_overall": float(df.max_abs_q_train.max()),
        "max_abs_q_by_investor": df.groupby("investor").max_abs_q_train.max().to_dict(),
        "all_full_rank": bool(df.full_rank.all()),
        "worst_cond": float(df.cond.max()),
        "min_sigma_min": float(df.sigma_min.min()),
        "all_lasso_unique": bool(df.lasso_unique.all()),
        "equicorr_check": df.dropna(subset=["equicorr_size"])
            .groupby("investor")[["equicorr_size", "equicorr_rank"]].max().to_dict()
            if "equicorr_size" in df else None,
        "k_star_max_saturated_by_investor": df.groupby("investor").k_star_max_saturated.max().to_dict(),
        "k_star_share_pct_by_investor": df.groupby("investor").k_star_share_pct.max().to_dict(),
        "full_saturation_lower_bound_min": df.groupby("investor").full_saturation_lower_bound.min().to_dict(),
        "F_hat_max_by_investor": df.groupby("investor").F_hat.max().to_dict(),
        "max_abs_action": float(df.max_abs_action.max()),
        "global_search_any_beaten": bool(sr.beaten.any()),
        "global_search_max_improvement": float(sr.improvement.max()),
        "global_search_total_starts": int(sr.n_starts.sum()),
    }
    (OUT / "prop11_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
