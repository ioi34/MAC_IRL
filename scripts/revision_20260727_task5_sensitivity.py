"""TASK 5: 망각계수 rho {0.95, 0.98, 0.99} x 모멘텀 창 {10, 20, 60} 민감도.

각 사양에서 원 파이프라인 그대로 특징을 재구축(raw -> prepare_daily_frame ->
build_state_feature_tensor -> 최신 973 유효행)하고, CPCV 45분할을 정확해로 재추정한다.
lambda는 현행 유형별 값(0 / 0.01 / 0.0003), 3특징 주 사양. 산출: 유형별 beta 부호 요약.

검증: (rho=0.98, window=20)이 기존 dataset_continuous_reward3.npz 를 재현하는지 대조.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.revision_20260727_core import LAMBDA_CFG, build_design, fit_exact, spec_terms
from src.data.contexts import build_context_matrix
from src.data.continuous_labels import add_continuous_action_labels, continuous_actions_array
from src.data.loaders import load_daily_frame
from src.data.preprocess import prepare_daily_frame
from src.data.splits import build_cpcv, combine_test_folds
from src.features.continuous import build_state_feature_tensor, valid_rows_for_continuous_model
from src.features.scaling import (
    fit_context_scaler,
    fit_feature_scaler,
    transform_context_matrix,
    transform_feature_tensor,
)
from src.utils.config import load_configs

OUT = Path("runs/revision_20260727")
OUT.mkdir(parents=True, exist_ok=True)

RHOS = [0.95, 0.98, 0.99]
WINDOWS = [10, 20, 60]


def build_dataset(base_config, rho: float, window: int):
    config = copy.deepcopy(base_config)
    config["preprocess"]["average_cost_rho"] = rho
    config["features"]["params"]["momentum"]["window"] = window
    raw = load_daily_frame(config)
    daily = prepare_daily_frame(raw, config)
    labeled = add_continuous_action_labels(daily, config)
    features, feature_names = build_state_feature_tensor(labeled, config)
    contexts, context_names = build_context_matrix(labeled, config)
    dates = pd.to_datetime(labeled[config["columns"]["date"]])
    sample = config["sample"]
    in_period = (dates >= pd.Timestamp(sample["start"])) & (dates <= pd.Timestamp(sample["end"]))
    valid = valid_rows_for_continuous_model(labeled, features, config["investors"], contexts)
    eligible = np.flatnonzero(in_period.to_numpy() & valid)
    idx = eligible[-int(sample["size"]):]
    actions = continuous_actions_array(labeled.iloc[idx], config["investors"]).astype(np.float64)
    return (
        features[idx].astype(np.float64),
        contexts[idx].astype(np.float64),
        actions,
        dates.iloc[idx].reset_index(drop=True),
        feature_names,
        context_names,
        config,
    )


def run_spec(config, features, contexts, actions, investors, feature_names, context_names):
    names = spec_terms("feat3", feature_names, context_names)
    cv = build_cpcv(config)
    split_input = np.arange(len(features)).reshape(-1, 1)
    rows = []
    for split_id, (train_idx, test_folds) in enumerate(cv.split(split_input)):
        context_scaler = fit_context_scaler(contexts, train_idx)
        scaled_contexts = transform_context_matrix(contexts, context_scaler).astype(np.float64)
        for inv_idx, investor in enumerate(investors):
            scaler = fit_feature_scaler(features[:, inv_idx], train_idx)
            scaled = transform_feature_tensor(features[:, inv_idx], scaler).astype(np.float64)
            design = build_design(scaled, scaled_contexts, 3)
            coef = fit_exact(design[train_idx], actions[train_idx, inv_idx], LAMBDA_CFG[investor])
            for name, value in zip(names, coef, strict=True):
                rows.append({"split": split_id, "investor": investor, "term": name, "weight": float(value)})
    return pd.DataFrame(rows)


def main() -> None:
    base_config = load_configs(
        "configs/data_continuous.yaml",
        "configs/features_continuous_reward3.yaml",
        "configs/model.yaml",
        "configs/train.yaml",
        "configs/experiment_continuous_reward3_ctxmain_default.yaml",
    )
    # 검증: 기준 사양 재구축이 기존 npz 재현하는지
    f, c, a, dates, fn, cn, _ = build_dataset(base_config, 0.98, 20)
    saved = np.load(base_config["paths"]["processed_dataset"], allow_pickle=False)
    dev_f = float(np.abs(f - saved["features"].astype(np.float64)).max())
    dev_a = float(np.abs(a - saved["actions"].astype(np.float64)).max())
    same_dates = bool((dates.dt.strftime("%Y-%m-%d").to_numpy() == saved["dates"].astype(str)).all())
    print(f"[verify] baseline rebuild: max|dF|={dev_f:.2e} max|dA|={dev_a:.2e} dates_match={same_dates}")

    all_rows = []
    for rho in RHOS:
        for window in WINDOWS:
            f, c, a, dates, fn, cn, config = build_dataset(base_config, rho, window)
            res = run_spec(config, f, c, a, list(config["investors"]), fn, cn)
            res["rho"] = rho
            res["momentum_window"] = window
            res["date_start"] = str(dates.iloc[0].date())
            res["date_end"] = str(dates.iloc[-1].date())
            all_rows.append(res)
            print(f"[TASK5] rho={rho} window={window} done ({dates.iloc[0].date()}..{dates.iloc[-1].date()})")
    res = pd.concat(all_rows, ignore_index=True)
    res.to_csv(OUT / "task5_sensitivity_coefficients.csv", index=False)
    summary = (
        res.groupby(["rho", "momentum_window", "investor", "term"])["weight"]
        .agg(mean="mean",
             sign_consistency=lambda s: float(max((s > 0).mean(), (s < 0).mean())),
             positive_rate=lambda s: float((s > 0).mean()),
             zero_rate=lambda s: float((s.abs() < 1e-12).mean()))
        .reset_index()
    )
    summary.to_csv(OUT / "task5_sensitivity_summary.csv", index=False)
    print("[done]")


if __name__ == "__main__":
    main()
