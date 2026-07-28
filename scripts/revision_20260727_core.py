"""2026-07-27 개정 실험 핵심 스크립트 (TASK 0 / TASK 1 / TASK 4).

TASK 0  정확해(좌표하강 Lasso) vs Adam 계수 대조 — 주 사양(3특징, 현행 λ).
TASK 1  지연 자기행동(u_{i,t})을 4번째 특징으로 승격한 사양의 V1–V4 전체 적용.
        - spec A (주보고): beta 4개 + B 6개(기존 3특징×2컨텍스트만) + alpha 2개 = 12 파라미터
        - spec B (부록):   beta 4개 + B 8개(지연 자기행동도 상호작용)   + alpha 2개 = 14 파라미터
TASK 4  세 유형 λ 통일 {0, 0.0005, 0.01} 강건성 재추정 (3특징 주 사양 + 4특징 spec A).

설계 원칙
- 목적식 (6): (1/n)||a - Z theta||^2 + lambda*||theta||_1, 절편 없음.
  sklearn Lasso 목적식은 (1/(2n))||.||^2 + alpha*||.||_1 이므로 alpha = lambda/2.
- CPCV 분할·purge/embargo·유형별 train-only 표준화는 원 파이프라인과 동일
  (skfolio CombinatorialPurgedCV 10/2/1/5, 분할 순서 동일).
- 지연 자기행동의 시점: 처리행 t는 (상태 x_t, 라벨 a_{t+1}=u_{i,t+1})이므로,
  상태일 t에 이용 가능한 최신 자기행동은 u_{i,t} = actions[t-1] (t일 종가 후 공개).
  이는 지속성 기준선 a_{t+1} <- a_t 와 동일한 정보 집합이며,
  scripts/diagnose_optimizer_and_baselines.py 의 lagged_action 정의와 같다.
  4특징 사양에서는 다른 특징과 동일하게 학습 인덱스에서 표준화한다.
- 기존 수치는 덮어쓰지 않는다: 모든 산출물은 runs/revision_20260727/ 아래 신규 파일.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge

from src.data.splits import build_cpcv, combine_test_folds
from src.evaluation.continuous_metrics import evaluate_continuous_actions
from src.features.scaling import (
    fit_context_scaler,
    fit_feature_scaler,
    transform_context_matrix,
    transform_feature_tensor,
)
from src.utils.config import load_configs

OUT = Path("runs/revision_20260727")
OUT.mkdir(parents=True, exist_ok=True)

LAMBDA_CFG = {"foreign": 0.0, "institution": 0.01, "retail": 0.0003}
UNIFIED_LAMBDAS = [0.0, 0.0005, 0.01]
N_BOOT = 1000
BOOT_SEED = 42

ADAM_RUN = Path("runs/continuous_reward3_ctxmain_default")
EXACT_RUN = Path("runs/continuous_reward3_ctxmain_default_exact_config")


# ---------------------------------------------------------------- solver

def fit_exact(design: np.ndarray, target: np.ndarray, lambda_l1: float) -> np.ndarray:
    if lambda_l1 <= 0.0:
        model = LinearRegression(fit_intercept=False)
    else:
        model = Lasso(alpha=lambda_l1 / 2.0, fit_intercept=False, max_iter=200_000, tol=1e-10)
    model.fit(design, target)
    return np.asarray(model.coef_, dtype=float).ravel()


def fit_penalty(design: np.ndarray, target: np.ndarray, penalty: str, strength: float) -> np.ndarray:
    if penalty == "none" or strength <= 0.0:
        model = LinearRegression(fit_intercept=False)
    elif penalty == "l1":
        model = Lasso(alpha=strength / 2.0, fit_intercept=False, max_iter=200_000, tol=1e-10)
    elif penalty == "l2":
        model = Ridge(alpha=strength * len(design) / 2.0, fit_intercept=False)
    else:
        raise ValueError(penalty)
    model.fit(design, target)
    return np.asarray(model.coef_, dtype=float).ravel()


# ---------------------------------------------------------------- data & designs

def load_base():
    config = load_configs(
        "configs/data_continuous.yaml",
        "configs/features_continuous_reward3.yaml",
        "configs/model.yaml",
        "configs/train.yaml",
        "configs/experiment_continuous_reward3_ctxmain_default.yaml",
    )
    data = np.load(config["paths"]["processed_dataset"], allow_pickle=False)
    saved_features = data["feature_names"].astype(str).tolist()
    feature_names = list(config["features"]["selected"])
    features = data["features"][..., [saved_features.index(n) for n in feature_names]]
    context_names = list(config["model"]["context_names"])
    saved_contexts = data["context_names"].astype(str).tolist()
    contexts = data["contexts"][:, [saved_contexts.index(c) for c in context_names]]
    actions = data["actions"].astype(np.float64)
    dates = pd.to_datetime(pd.Series(data["dates"].astype(str)))
    investors = list(config["investors"])
    # 지연 자기행동 u_{i,t}: 행 t-1의 라벨 = u_{i,t}. 첫 행은 0.
    lagged = np.vstack([np.zeros((1, actions.shape[1])), actions[:-1]])
    return config, features, contexts, actions, dates, investors, feature_names, context_names, lagged


def term_names(feature_names, context_names, interacting, alpha=True):
    names = [f"beta:{f}" for f in feature_names]
    names += [f"B:{f}x{c}" for f in interacting for c in context_names]
    if alpha:
        names += [f"alpha:{c}" for c in context_names]
    return names


def build_design(scaled_features, scaled_contexts, n_interacting):
    """[x_1..x_F, (x_f*c_k) for f in 1..n_interacting, c_1..c_K]."""
    n_c = scaled_contexts.shape[1]
    inter = np.column_stack(
        [scaled_features[:, f] * scaled_contexts[:, k] for f in range(n_interacting) for k in range(n_c)]
    )
    return np.column_stack([scaled_features, inter, scaled_contexts]).astype(np.float64)


def make_feature_tensor(features, lagged, investors, with_lag: bool):
    """(n, n_inv, F) 텐서. with_lag이면 F=4 (마지막 열 = u_{i,t})."""
    if not with_lag:
        return features
    n, n_inv, _ = features.shape
    out = np.empty((n, n_inv, features.shape[2] + 1), dtype=np.float64)
    out[:, :, : features.shape[2]] = features
    for i in range(n_inv):
        out[:, i, features.shape[2]] = lagged[:, i]
    return out


SPECS = {
    # name: (with_lag, n_interacting_features, feature_names_fn)
    "feat3": (False, 3),
    "feat4_main": (True, 3),   # spec A: 지연 자기행동은 beta만 (상호작용 없음)
    "feat4_full": (True, 4),   # spec B: 지연 자기행동도 컨텍스트와 상호작용
}


def spec_feature_names(spec, feature_names):
    return feature_names + (["lag_own"] if SPECS[spec][0] else [])


def spec_terms(spec, feature_names, context_names):
    fnames = spec_feature_names(spec, feature_names)
    interacting = fnames[: SPECS[spec][1]]
    return term_names(fnames, context_names, interacting)


# ---------------------------------------------------------------- V1: CPCV

def run_cpcv(spec, config, features, contexts, actions, investors, feature_names, context_names,
             lagged, lambdas):
    with_lag, n_int = SPECS[spec]
    tensor = make_feature_tensor(features, lagged, investors, with_lag)
    names = spec_terms(spec, feature_names, context_names)
    cv = build_cpcv(config)
    split_input = np.arange(len(tensor)).reshape(-1, 1)

    coef_rows, metric_rows = [], []
    for split_id, (train_idx, test_folds) in enumerate(cv.split(split_input)):
        test_idx = combine_test_folds(test_folds)
        context_scaler = fit_context_scaler(contexts, train_idx)
        scaled_contexts = transform_context_matrix(contexts, context_scaler).astype(np.float64)
        for inv_idx, investor in enumerate(investors):
            scaler = fit_feature_scaler(tensor[:, inv_idx], train_idx)
            scaled = transform_feature_tensor(tensor[:, inv_idx], scaler).astype(np.float64)
            design = build_design(scaled, scaled_contexts, n_int)
            y = actions[:, inv_idx]
            lam = lambdas[investor]
            coef = fit_exact(design[train_idx], y[train_idx], lam)

            q_train = design[train_idx] @ coef
            q_test = design[test_idx] @ coef
            pred = np.clip(q_test, -1.0, 1.0)
            metrics = evaluate_continuous_actions(y[test_idx], pred)
            persistence = np.concatenate([[0.0], y[:-1]])[test_idx]
            base = evaluate_continuous_actions(y[test_idx], persistence)
            denom = float(np.mean((y[test_idx] - y[train_idx].mean()) ** 2))
            metric_rows.append({
                "spec": spec, "split": split_id, "investor": investor, "lambda_l1": lam,
                "direction_accuracy": metrics["direction_accuracy"],
                "correlation": metrics["correlation"], "mae": metrics["mae"],
                "oos_r2": 1.0 - metrics["mse"] / denom if denom else np.nan,
                "saturation_rate": metrics["saturation_rate"],
                "train_saturation_rate": float((np.abs(q_train) >= 1.0).mean()),
                "max_abs_q": float(max(np.abs(q_train).max(), np.abs(q_test).max())),
                "persistence_direction_accuracy": base["direction_accuracy"],
                "persistence_correlation": base["correlation"],
                "persistence_mae": base["mae"],
            })
            for name, value in zip(names, coef, strict=True):
                coef_rows.append({
                    "spec": spec, "split": split_id, "investor": investor,
                    "term": name, "weight": float(value),
                })
    return pd.DataFrame(coef_rows), pd.DataFrame(metric_rows)


def summarize_coefs(coefs: pd.DataFrame) -> pd.DataFrame:
    def agg(s):
        return pd.Series({
            "mean": s.mean(), "std": s.std(),
            "positive_rate": float((s > 0).mean()),
            "negative_rate": float((s < 0).mean()),
            "zero_rate": float((s.abs() < 1e-12).mean()),
            "sign_consistency": float(max((s > 0).mean(), (s < 0).mean())),
        })
    return coefs.groupby(["spec", "investor", "term"])["weight"].apply(agg).unstack().reset_index()


# ---------------------------------------------------------------- V2: 달력월 블록 부트스트랩

def month_bootstrap(spec, features, contexts, actions, dates, investors, feature_names,
                    context_names, lagged, lambdas, n_boot=N_BOOT, seed=BOOT_SEED):
    """validate_protocol.calendar_month_bootstrap 과 동일한 방법론 (전체표본 스케일러, 월 단위 복원추출)."""
    with_lag, n_int = SPECS[spec]
    tensor = make_feature_tensor(features, lagged, investors, with_lag)
    names = spec_terms(spec, feature_names, context_names)
    rng = np.random.default_rng(seed)
    months = dates.dt.to_period("M").to_numpy()
    unique_months = np.unique(months)
    month_rows = {m: np.flatnonzero(months == m) for m in unique_months}

    rows = []
    all_idx = np.arange(len(tensor))
    for inv_idx, investor in enumerate(investors):
        scaler = fit_feature_scaler(tensor[:, inv_idx], all_idx)
        scaled = transform_feature_tensor(tensor[:, inv_idx], scaler).astype(np.float64)
        context_scaler = fit_context_scaler(contexts, all_idx)
        scaled_contexts = transform_context_matrix(contexts, context_scaler).astype(np.float64)
        design = build_design(scaled, scaled_contexts, n_int)
        y = actions[:, inv_idx]
        draws = np.empty((n_boot, design.shape[1]))
        for b in range(n_boot):
            picked = rng.choice(unique_months, size=len(unique_months), replace=True)
            idx = np.concatenate([month_rows[m] for m in picked])
            draws[b] = fit_exact(design[idx], y[idx], lambdas[investor])
        for j, name in enumerate(names):
            col = draws[:, j]
            lo, hi = np.percentile(col, [2.5, 97.5])
            rows.append({
                "spec": spec, "investor": investor, "term": name,
                "mean": float(col.mean()), "ci_lower": float(lo), "ci_upper": float(hi),
                "sign_consistency": float(max((col > 0).mean(), (col < 0).mean())),
                "excludes_zero": bool(lo > 0 or hi < 0),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- V3: expanding walk-forward

def walk_forward(spec, features, contexts, actions, dates, investors, feature_names,
                 context_names, lagged, lambdas, initial=400, step=60, purge=5):
    with_lag, n_int = SPECS[spec]
    tensor = make_feature_tensor(features, lagged, investors, with_lag)
    names = spec_terms(spec, feature_names, context_names)
    n = len(tensor)
    rows = []
    for inv_idx, investor in enumerate(investors):
        y = actions[:, inv_idx]
        origin = 0
        while True:
            train_end = initial + origin * step
            test_start = train_end + purge
            test_end = min(test_start + step, n)
            if test_end - test_start < 10:
                break
            train_idx = np.arange(train_end)
            test_idx = np.arange(test_start, test_end)
            scaler = fit_feature_scaler(tensor[:, inv_idx], train_idx)
            scaled = transform_feature_tensor(tensor[:, inv_idx], scaler).astype(np.float64)
            context_scaler = fit_context_scaler(contexts, train_idx)
            scaled_contexts = transform_context_matrix(contexts, context_scaler).astype(np.float64)
            design = build_design(scaled, scaled_contexts, n_int)
            coef = fit_exact(design[train_idx], y[train_idx], lambdas[investor])
            pred = np.clip(design[test_idx] @ coef, -1.0, 1.0)
            metrics = evaluate_continuous_actions(y[test_idx], pred)
            persistence = np.concatenate([[0.0], y[:-1]])[test_idx]
            base = evaluate_continuous_actions(y[test_idx], persistence)
            row = {
                "spec": spec, "investor": investor, "window": origin,
                "train_end_date": str(dates.iloc[train_end - 1].date()),
                "test_start_date": str(dates.iloc[test_start].date()),
                "test_end_date": str(dates.iloc[test_end - 1].date()),
                "direction_accuracy": metrics["direction_accuracy"],
                "correlation": metrics["correlation"],
                "persistence_direction_accuracy": base["direction_accuracy"],
            }
            row.update({name: float(v) for name, v in zip(names, coef, strict=True)})
            rows.append(row)
            origin += 1
            if initial + origin * step + purge >= n:
                break
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- V4: 벌점 대조

def penalty_swap(spec, config, features, contexts, actions, investors, feature_names,
                 context_names, lagged, lambdas):
    with_lag, n_int = SPECS[spec]
    tensor = make_feature_tensor(features, lagged, investors, with_lag)
    names = spec_terms(spec, feature_names, context_names)
    cv = build_cpcv(config)
    split_input = np.arange(len(tensor)).reshape(-1, 1)
    rows = []
    for split_id, (train_idx, test_folds) in enumerate(cv.split(split_input)):
        context_scaler = fit_context_scaler(contexts, train_idx)
        scaled_contexts = transform_context_matrix(contexts, context_scaler).astype(np.float64)
        for inv_idx, investor in enumerate(investors):
            scaler = fit_feature_scaler(tensor[:, inv_idx], train_idx)
            scaled = transform_feature_tensor(tensor[:, inv_idx], scaler).astype(np.float64)
            design = build_design(scaled, scaled_contexts, n_int)
            y = actions[:, inv_idx]
            for penalty, strength in (
                ("l1", lambdas[investor]),
                ("l2", max(lambdas[investor], 0.001)),
                ("none", 0.0),
            ):
                coef = fit_penalty(design[train_idx], y[train_idx], penalty, strength)
                row = {"spec": spec, "split": split_id, "investor": investor, "penalty": penalty}
                row.update({n_: float(v) for n_, v in zip(names, coef, strict=True)})
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- TASK 0: Adam vs 정확해

def load_adam_split_weights(feature_names, context_names):
    beta = pd.read_csv(ADAM_RUN / "reward_weights.csv")
    bmat = pd.read_csv(ADAM_RUN / "context_weights.csv")
    alpha = pd.read_csv(ADAM_RUN / "context_main_weights.csv")
    rows = []
    for _, r in beta.iterrows():
        rows.append({"split": r["split"], "investor": r["investor"],
                     "term": f"beta:{r['feature']}", "weight": r["weight"]})
    for _, r in bmat.iterrows():
        rows.append({"split": r["split"], "investor": r["investor"],
                     "term": f"B:{r['feature']}x{r['context']}", "weight": r["weight"]})
    for _, r in alpha.iterrows():
        rows.append({"split": r["split"], "investor": r["investor"],
                     "term": f"alpha:{r['context']}", "weight": r["weight"]})
    return pd.DataFrame(rows)


def task0(config, features, contexts, actions, investors, feature_names, context_names, lagged,
          exact_coefs_feat3):
    adam = load_adam_split_weights(feature_names, context_names)
    exact = exact_coefs_feat3.rename(columns={"weight": "weight_exact"})[
        ["split", "investor", "term", "weight_exact"]
    ]
    merged = adam.rename(columns={"weight": "weight_adam"}).merge(
        exact, on=["split", "investor", "term"], how="inner", validate="one_to_one"
    )
    merged["abs_diff"] = (merged.weight_adam - merged.weight_exact).abs()

    # 유형별·항목별 대조 (45분할 평균 기준 + 분할 수준 최대 격차)
    rows = []
    for (investor, term), g in merged.groupby(["investor", "term"], sort=False):
        m_adam, m_exact = g.weight_adam.mean(), g.weight_exact.mean()
        rows.append({
            "investor": investor, "term": term,
            "adam_mean": m_adam, "exact_mean": m_exact,
            "diff_mean": m_adam - m_exact,
            "match_4dp_mean": bool(round(m_adam, 4) == round(m_exact, 4)),
            "share_splits_match_4dp": float((g.weight_adam.round(4) == g.weight_exact.round(4)).mean()),
            "max_abs_diff_across_splits": float(g.abs_diff.max()),
            "adam_sign_consistency": float(max((g.weight_adam > 0).mean(), (g.weight_adam < 0).mean())),
            "exact_sign_consistency": float(max((g.weight_exact > 0).mean(), (g.weight_exact < 0).mean())),
            "exact_zero_rate": float((g.weight_exact.abs() < 1e-12).mean()),
            "adam_zero_rate": float((g.weight_adam.abs() < 1e-12).mean()),
        })
    table = pd.DataFrame(rows)

    # 목적함수 격차 (분할별): Adam 해 vs 정확해, 동일 목적식으로 재계산
    cv = build_cpcv(config)
    split_input = np.arange(len(features)).reshape(-1, 1)
    obj_rows = []
    adam_piv = adam.pivot_table(index=["split", "investor"], columns="term", values="weight")
    exact_piv = exact_coefs_feat3.pivot_table(index=["split", "investor"], columns="term", values="weight")
    names = spec_terms("feat3", feature_names, context_names)
    for split_id, (train_idx, test_folds) in enumerate(cv.split(split_input)):
        context_scaler = fit_context_scaler(contexts, train_idx)
        scaled_contexts = transform_context_matrix(contexts, context_scaler).astype(np.float64)
        for inv_idx, investor in enumerate(investors):
            scaler = fit_feature_scaler(features[:, inv_idx], train_idx)
            scaled = transform_feature_tensor(features[:, inv_idx], scaler).astype(np.float64)
            design = build_design(scaled, scaled_contexts, 3)[train_idx]
            y = actions[train_idx, inv_idx]
            lam = LAMBDA_CFG[investor]

            def objective(w):
                pred = np.clip(design @ w, -1.0, 1.0)
                return float(np.mean((pred - y) ** 2) + lam * np.abs(w).sum())

            w_adam = adam_piv.loc[(split_id, investor), names].to_numpy(dtype=float)
            w_exact = exact_piv.loc[(split_id, investor), names].to_numpy(dtype=float)
            o_adam, o_exact = objective(w_adam), objective(w_exact)
            obj_rows.append({
                "split": split_id, "investor": investor,
                "objective_adam": o_adam, "objective_exact": o_exact,
                "relative_gap_pct": 100 * (o_adam - o_exact) / o_exact,
                "l1_norm_adam": float(np.abs(w_adam).sum()),
                "l1_norm_exact": float(np.abs(w_exact).sum()),
            })
    return table, pd.DataFrame(obj_rows), merged


# ---------------------------------------------------------------- main

def main() -> None:
    config, features, contexts, actions, dates, investors, feature_names, context_names, lagged = load_base()

    # --- 재현성 검증: 3특징 정확해 CPCV를 재계산해 저장본과 대조
    coefs3, metrics3 = run_cpcv("feat3", config, features, contexts, actions, investors,
                                feature_names, context_names, lagged, LAMBDA_CFG)
    saved = pd.read_csv(EXACT_RUN / "reward_weights.csv")
    saved["term"] = "beta:" + saved.feature
    check = coefs3[coefs3.term.str.startswith("beta:")].merge(
        saved, on=["split", "investor", "term"], suffixes=("_new", "_saved"))
    max_dev = float((check.weight_new - check.weight_saved).abs().max())
    print(f"[verify] 3특징 정확해 재계산 vs 저장본 최대 편차: {max_dev:.2e}")

    # --- TASK 0
    t0_table, t0_obj, t0_split_level = task0(
        config, features, contexts, actions, investors, feature_names, context_names, lagged, coefs3)
    t0_table.to_csv(OUT / "task0_adam_vs_exact_by_term.csv", index=False)
    t0_obj.to_csv(OUT / "task0_objective_gap_by_split.csv", index=False)
    t0_split_level.to_csv(OUT / "task0_adam_vs_exact_split_level.csv", index=False)
    print("[TASK0] saved")

    # --- TASK 1: V1
    all_coefs = [coefs3]
    all_metrics = [metrics3]
    for spec in ("feat4_main", "feat4_full"):
        c, m = run_cpcv(spec, config, features, contexts, actions, investors,
                        feature_names, context_names, lagged, LAMBDA_CFG)
        all_coefs.append(c)
        all_metrics.append(m)
    coefs = pd.concat(all_coefs, ignore_index=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    coefs.to_csv(OUT / "task1_cpcv_coefficients.csv", index=False)
    metrics.to_csv(OUT / "task1_cpcv_metrics.csv", index=False)
    summarize_coefs(coefs).to_csv(OUT / "task1_cpcv_coefficients_summary.csv", index=False)
    msum = metrics.groupby(["spec", "investor"]).agg(
        direction_accuracy=("direction_accuracy", "mean"),
        direction_accuracy_std=("direction_accuracy", "std"),
        correlation=("correlation", "mean"),
        correlation_std=("correlation", "std"),
        mae=("mae", "mean"),
        oos_r2=("oos_r2", "mean"),
        saturation_rate=("saturation_rate", "max"),
        train_saturation_rate=("train_saturation_rate", "max"),
        max_abs_q=("max_abs_q", "max"),
        persistence_direction_accuracy=("persistence_direction_accuracy", "mean"),
        persistence_correlation=("persistence_correlation", "mean"),
        persistence_mae=("persistence_mae", "mean"),
    ).reset_index()
    msum.to_csv(OUT / "task1_cpcv_metrics_summary.csv", index=False)
    print("[TASK1-V1] saved")

    # --- TASK 1: V2 부트스트랩 (4특징 두 버전; 3특징은 runs/protocol_validation 재사용)
    boots = []
    for spec in ("feat4_main", "feat4_full"):
        boots.append(month_bootstrap(spec, features, contexts, actions, dates, investors,
                                     feature_names, context_names, lagged, LAMBDA_CFG))
        print(f"[TASK1-V2] {spec} bootstrap done")
    boot = pd.concat(boots, ignore_index=True)
    boot.to_csv(OUT / "task1_month_bootstrap.csv", index=False)

    # --- TASK 1: V3 walk-forward
    wfs = [walk_forward(spec, features, contexts, actions, dates, investors,
                        feature_names, context_names, lagged, LAMBDA_CFG)
           for spec in ("feat3", "feat4_main", "feat4_full")]
    wf = pd.concat(wfs, ignore_index=True)
    wf.to_csv(OUT / "task1_walk_forward.csv", index=False)
    print("[TASK1-V3] saved")

    # --- TASK 1: V4 벌점 대조
    swaps = [penalty_swap(spec, config, features, contexts, actions, investors,
                          feature_names, context_names, lagged, LAMBDA_CFG)
             for spec in ("feat4_main", "feat4_full")]
    swap = pd.concat(swaps, ignore_index=True)
    swap.to_csv(OUT / "task1_penalty_swap.csv", index=False)
    id_cols = ["spec", "split", "investor", "penalty"]
    swap_long = swap.melt(id_vars=id_cols, var_name="term", value_name="weight").dropna()
    swap_summary = (
        swap_long.groupby(["spec", "investor", "penalty", "term"])["weight"]
        .agg(mean="mean",
             sign_consistency=lambda s: float(max((s > 0).mean(), (s < 0).mean())),
             zero_rate=lambda s: float((s.abs() < 1e-12).mean()))
        .reset_index()
    )
    swap_summary.to_csv(OUT / "task1_penalty_swap_summary.csv", index=False)
    print("[TASK1-V4] saved")

    # --- TASK 4: 통일 λ
    uni_coefs, uni_metrics = [], []
    for lam in UNIFIED_LAMBDAS:
        lam_map = {inv: lam for inv in investors}
        for spec in ("feat3", "feat4_main"):
            c, m = run_cpcv(spec, config, features, contexts, actions, investors,
                            feature_names, context_names, lagged, lam_map)
            c["unified_lambda"] = lam
            m["unified_lambda"] = lam
            uni_coefs.append(c)
            uni_metrics.append(m)
        print(f"[TASK4] lambda={lam} done")
    uc = pd.concat(uni_coefs, ignore_index=True)
    um = pd.concat(uni_metrics, ignore_index=True)
    uc.to_csv(OUT / "task4_unified_lambda_coefficients.csv", index=False)
    um.to_csv(OUT / "task4_unified_lambda_metrics.csv", index=False)
    usum = (
        uc.groupby(["spec", "unified_lambda", "investor", "term"])["weight"]
        .agg(mean="mean",
             sign_consistency=lambda s: float(max((s > 0).mean(), (s < 0).mean())),
             zero_rate=lambda s: float((s.abs() < 1e-12).mean()))
        .reset_index()
    )
    usum.to_csv(OUT / "task4_unified_lambda_summary.csv", index=False)

    (OUT / "run_info.json").write_text(json.dumps({
        "verify_max_dev_vs_saved_exact": max_dev,
        "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
        "lambda_config": LAMBDA_CFG, "unified_lambdas": UNIFIED_LAMBDAS,
        "lag_definition": "u_{i,t} = actions[t-1] (state-date t, published after close t); scaled with other features on train indices",
    }, indent=2))
    print("[done]", OUT)


if __name__ == "__main__":
    main()
