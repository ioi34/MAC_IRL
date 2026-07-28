"""TASK 2 보강: 조건부 청산 null 시뮬레이션.

외국인만 선호를 갖고(모멘텀 +0.0502, alpha_KOSPI +0.0360, alpha_FX -0.0269;
정확해 CPCV 평균), 개인의 직접 특징계수는 0인 세계를 생성한다. 외국인 신호의
금액 반응은 관측 흡수율로 기관·개인·기타에 배분한다. 세 투자자 유형의 특징 비관련
금액 잔차는 네 주체를 한 묶음으로 원형 이동해 특징과의 정렬만 끊는다. 이 공동 이동은
일별 청산식, 거래규모, 잔차의 시간순서와 횡단면 관계를 보존한다. 잔차 전체에 하나의
공통 척도를 적용해 합성 개인 SD를 관측치에 맞추며, 공통 척도이므로 청산식은 유지된다.

동일 파이프라인으로 세 유형을 재추정해 '개인이 직접 선호 없이 물려받는' 계수의
조건부 분포를 얻는다. 1차 근사 산술(task2_induced_vs_observed)의 파이프라인 검증판이다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.revision_20260727_core import LAMBDA_CFG, build_design, fit_exact, spec_terms
from scripts.revision_20260727_task2_clearing import (
    INVESTORS,
    fit_ar1,
    load_raw_aligned,
    part1_arithmetic,
)
from src.features.scaling import (
    fit_context_scaler,
    fit_feature_scaler,
    transform_context_matrix,
    transform_feature_tensor,
)

OUT = Path("runs/revision_20260727")
N_SIM = 1000
SEED = 4242

FOREIGN_THETA = {"momentum": 0.0502, "kospi_return_1d": 0.0360, "fx_level_z_252": -0.0269}
OBSERVED_RETAIL = {
    "beta:momentum": -0.0383,
    "alpha:kospi_return_1d": -0.0238,
    "alpha:fx_level_z_252": 0.0355,
}


def common_residual_scale(
    signal: np.ndarray,
    residual_net: np.ndarray,
    weights: np.ndarray,
    phi_retail: float,
    target_sd: float,
) -> float:
    """모든 금액잔차에 같은 척도를 적용해 이동 평균 합성 개인 분산을 관측치에 맞춘다."""
    a = b = c = 0.0
    n = len(signal)
    for shift in range(1, n):
        shifted_w = np.roll(weights, shift, axis=0)
        noise = np.roll(residual_net, shift, axis=0)[:, 2] / shifted_w[:, 2]
        induced = phi_retail * signal * shifted_w[:, 0] / shifted_w[:, 2]
        noise = noise - noise.mean()
        induced = induced - induced.mean()
        a += float(np.mean(noise ** 2))
        b += float(2.0 * np.mean(noise * induced))
        c += float(np.mean(induced ** 2))
    a, b, c = a / (n - 1), b / (n - 1), c / (n - 1)
    roots = np.roots([a, b, c - target_sd ** 2])
    positive = [float(r.real) for r in roots if abs(r.imag) < 1e-10 and r.real > 0]
    if len(positive) != 1:
        raise AssertionError("개인 SD 보정 척도의 양의 실근이 유일하지 않음")
    return positive[0]


def main() -> None:
    _, raw, data, _, label_pos = load_raw_aligned()
    rng = np.random.default_rng(SEED)
    lab = raw.iloc[label_pos].reset_index(drop=True)
    _, absorb, _, _ = part1_arithmetic(raw, label_pos)
    phi_i = float(absorb["foreign"]["institution"])
    phi_r = float(absorb["foreign"]["retail"])
    phi_o = float(absorb["foreign"]["other(resid)"])
    if not np.isclose(phi_i + phi_r + phi_o, -1.0, atol=1e-10):
        raise AssertionError("외국인 흡수율의 합이 -1이 아님")

    features = data["features"].astype(np.float64)
    contexts = data["contexts"].astype(np.float64)
    saved_feature_names = data["feature_names"].astype(str).tolist()
    fi_mom = saved_feature_names.index("momentum")
    fi_herd = saved_feature_names.index("herd")
    fi_uw = saved_feature_names.index("underwater")

    # 외국인 DGP 회귀자: 상태일(t)의 표준화 momentum·컨텍스트가 라벨일(t+1)의 u_f를 결정
    x_mom = features[:, 0, fi_mom]
    x_mom_z = (x_mom - x_mom.mean()) / x_mom.std(ddof=0)
    ctx_z = (contexts - contexts.mean(0)) / contexts.std(0, ddof=0)
    signal_lab = (FOREIGN_THETA["momentum"] * x_mom_z
                  + FOREIGN_THETA["kospi_return_1d"] * ctx_z[:, 0]
                  + FOREIGN_THETA["fx_level_z_252"] * ctx_z[:, 1])
    signal_lab = signal_lab - signal_lab.mean()

    weights = np.column_stack([lab[f"W_{inv}"].to_numpy() for inv in INVESTORS])
    trading_value = lab["trading_value"].to_numpy()
    observed_net = np.column_stack([
        lab["net_foreign"], lab["net_institution"], lab["net_retail"], lab["resid"],
    ])
    phi = np.array([1.0, phi_i, phi_r, phi_o])
    signal_net_f = signal_lab * weights[:, 0]
    residual_net = observed_net - signal_net_f[:, None] * phi[None, :]
    if np.max(np.abs(residual_net.sum(axis=1))) > 1e-3:
        raise AssertionError("관측 금액잔차의 청산식 위반")

    targets = {}
    for inv in INVESTORS:
        observed = lab[f"u_{inv}"].to_numpy()
        targets[inv] = {
            "mean": float(observed.mean()),
            "sd": float(observed.std(ddof=0)),
            "ar1": fit_ar1(observed),
        }
    targets["other"] = {
        "sd_KRW": float(lab["resid"].std(ddof=0)),
    }
    residual_scale = common_residual_scale(
        signal_lab, residual_net, weights, phi_r, targets["retail"]["sd"],
    )

    names3 = spec_terms("feat3", ["momentum", "herd", "underwater"], ["kospi_return_1d", "fx_level_z_252"])

    rows, cal_rows = [], []
    shifts = rng.choice(np.arange(1, len(lab)), size=N_SIM, replace=N_SIM > len(lab) - 1)
    for s, shift in enumerate(shifts):
        shifted_weights = np.roll(weights, shift, axis=0)
        shifted_tv = np.roll(trading_value, shift)
        shifted_residual = residual_scale * np.roll(residual_net, shift, axis=0)
        signal_net = signal_lab * shifted_weights[:, 0]
        net_syn = signal_net[:, None] * phi[None, :] + shifted_residual
        if np.max(np.abs(net_syn.sum(axis=1))) > 1e-3:
            raise AssertionError("합성 수급의 금액 청산식 위반")
        u_matrix = net_syn[:, :3] / shifted_weights
        u_mkt = net_syn[:, :3] / shifted_tv[:, None]
        cal_rows.append({
            "sim": s, "shift": int(shift),
            "foreign_sd": float(u_matrix[:, 0].std(ddof=0)),
            "foreign_ar1": fit_ar1(u_matrix[:, 0]),
            "institution_sd": float(u_matrix[:, 1].std(ddof=0)),
            "institution_ar1": fit_ar1(u_matrix[:, 1]),
            "retail_sd": float(u_matrix[:, 2].std(ddof=0)),
            "retail_ar1": fit_ar1(u_matrix[:, 2]),
            "corr_f_r": float(np.corrcoef(u_matrix[:, 0], u_matrix[:, 2])[0, 1]),
            "corr_i_r": float(np.corrcoef(u_matrix[:, 1], u_matrix[:, 2])[0, 1]),
            "other_sd_KRW": float(net_syn[:, 3].std(ddof=0)),
        })

        for inv_idx, inv in enumerate(INVESTORS):
            others = [o for o in range(len(INVESTORS)) if o != inv_idx]
            herd = np.empty(len(lab))
            herd[:2] = features[:2, inv_idx, fi_herd]
            herd[2:] = 0.5 * (u_mkt[:-2, others[0]] + u_mkt[:-2, others[1]])
            X = np.column_stack([features[:, inv_idx, fi_mom], herd, features[:, inv_idx, fi_uw]])
            y = u_matrix[:, inv_idx]
            all_idx = np.arange(len(X))
            scaler = fit_feature_scaler(X[:, None, :], all_idx)
            scaled = transform_feature_tensor(X[:, None, :], scaler)[:, 0, :].astype(np.float64)
            cs = fit_context_scaler(contexts, all_idx)
            sc = transform_context_matrix(contexts, cs).astype(np.float64)
            design = build_design(scaled, sc, 3)
            coef = fit_exact(design, y, LAMBDA_CFG[inv])
            for name, value in zip(names3, coef, strict=True):
                rows.append({"sim": s, "investor": inv, "term": name, "weight": float(value)})

    sims = pd.DataFrame(rows)
    sims.to_csv(OUT / "task2_conditional_null_coefficients.csv", index=False)
    q = (
        sims.groupby(["investor", "term"])["weight"]
        .agg(mean="mean", sd="std",
             q2_5=lambda x: float(np.percentile(x, 2.5)),
             q50=lambda x: float(np.percentile(x, 50)),
             q97_5=lambda x: float(np.percentile(x, 97.5)))
        .reset_index()
    )
    q.to_csv(OUT / "task2_conditional_null_summary.csv", index=False)
    calibration = pd.DataFrame(cal_rows)
    calibration.to_csv(OUT / "task2_conditional_null_calibration.csv", index=False)

    comparison_rows = []
    for term, observed in OBSERVED_RETAIL.items():
        values = sims[(sims.investor == "retail") & (sims.term == term)].weight
        lo, hi = np.percentile(values, [2.5, 97.5])
        comparison_rows.append({
            "term": term,
            "observed": observed,
            "null_mean": float(values.mean()),
            "null_sd": float(values.std(ddof=1)),
            "q2_5": float(lo),
            "q97_5": float(hi),
            "percentile": float((values <= observed).mean()),
            "outside_95": bool(observed < lo or observed > hi),
        })
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUT / "task2_conditional_null_comparison.csv", index=False)

    info = {
        "foreign_theta_dgp": FOREIGN_THETA,
        "absorption_slopes": {"institution": phi_i, "retail": phi_r, "other": phi_o},
        "targets": targets,
        "residual_resampling": "joint circular shift in net-flow space",
        "residual_common_scale": residual_scale,
        "mean_calibration": calibration.mean(numeric_only=True).to_dict(),
    }
    with open(OUT / "task2_conditional_null_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(q.round(4).to_string(index=False))
    print(comparison.round(4).to_string(index=False))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
