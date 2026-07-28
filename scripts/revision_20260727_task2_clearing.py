"""TASK 2: 시장 청산 제약이 유형 간 계수에 기계적으로 유도하는 성분의 정량화.

파트 1 (산술): 기존 973 표본의 일별 유형별 총거래금액 W_{i,t}(매수+매도)와
순매수금액으로부터, 외국인 계수가 청산 제약을 통해 개인 계수에 유도하는 성분을
1차 근사로 계산한다.

유도 (가정 명시):
  (A1) u_{i,t} = net_i,t / W_i,t  (유형별 총거래금액 정규화, 식 (1)과 동일).
  (A2) 청산: net_f + net_inst + net_r + resid = 0. resid는 기타법인·기타외국인 등.
  (A3) 외국인이 표준화 특징 x에 beta_f 만큼 반응하면 순매수금액 반응은
       d net_f = beta_f * W_f (W는 x와 독립적인 규모 변수로 취급 — 1차 근사).
  (A4) 그 금액 반응 중 개인이 흡수하는 몫 phi_r 은 일별 금액 회귀
       net_r = a + phi_r * net_f + e 의 기울기로 추정하며, x가 유발한 변동과
       그 외 변동의 흡수율이 같다고 가정한다(동질 흡수 가정).
  => beta_r^induced = phi_r * beta_f * E[W_f / W_r]
     가중 방식: (a) 일별 비율의 단순평균 E[W_f/W_r], (b) 거래대금 가중 sum(W_f)/sum(W_r),
     (c) 중앙값. 세 값으로 범위를 제시한다.

파트 2 (합성 null, 부록): 선호가 전혀 없고 청산 상쇄만 있는 합성 수급을 생성해
동일 파이프라인(전체표본 표준화, 정확해)으로 계수를 추정, 기계적 계수의 분포를 얻는다.
  - u_f, u_inst: 관측 SD·AR(1)에 맞춘 가우시안 AR(1) (특징과 독립).
  - resid: 실측 resid의 평균·SD·AR(1)에 맞춘 AR(1).
  - net_r = -(net_f + net_inst + resid), u_r = net_r / W_r (실측 W 사용).
  - momentum·underwater·컨텍스트는 실측(가격·거래 기반, u와 무관), herd·지연자기행동·라벨은 합성 u로 재구성.
  - 관측 대비: 합성 분포에서 관측 |beta|를 넘는 비율, 동시점 상관 재현 여부.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.revision_20260727_core import build_design, fit_exact, spec_terms, LAMBDA_CFG
from src.features.scaling import (
    fit_context_scaler,
    fit_feature_scaler,
    transform_context_matrix,
    transform_feature_tensor,
)
from src.utils.config import load_configs

OUT = Path("runs/revision_20260727")
OUT.mkdir(parents=True, exist_ok=True)
N_SIM = 1000
SEED = 42

INVESTORS = ["foreign", "institution", "retail"]


def load_raw_aligned():
    config = load_configs(
        "configs/data_continuous.yaml",
        "configs/features_continuous_reward3.yaml",
        "configs/model.yaml",
        "configs/train.yaml",
        "configs/experiment_continuous_reward3_ctxmain_default.yaml",
    )
    raw = pd.read_csv(config["paths"]["raw_daily"])
    raw.columns = raw.columns.str.strip()
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values("date").reset_index(drop=True)
    for inv in INVESTORS:
        raw[f"net_{inv}"] = raw[f"{inv}_buy_value"] - raw[f"{inv}_sell_value"]
        raw[f"W_{inv}"] = raw[f"{inv}_buy_value"] + raw[f"{inv}_sell_value"]
        raw[f"u_{inv}"] = raw[f"net_{inv}"] / raw[f"W_{inv}"]
    raw["resid"] = -(raw["net_foreign"] + raw["net_institution"] + raw["net_retail"])

    data = np.load(config["paths"]["processed_dataset"], allow_pickle=False)
    state_dates = pd.to_datetime(pd.Series(data["dates"].astype(str)))
    pos = raw.set_index("date").index.get_indexer(state_dates)
    assert (pos >= 0).all()
    label_pos = pos + 1  # 라벨 a_{t+1}의 달력일
    assert label_pos.max() < len(raw)
    return config, raw, data, pos, label_pos


def part1_arithmetic(raw, label_pos):
    lab = raw.iloc[label_pos]
    stats = {}
    ratios = {}
    for a in INVESTORS:
        for b in INVESTORS:
            if a == b:
                continue
            r = lab[f"W_{a}"] / lab[f"W_{b}"]
            ratios[f"{a}/{b}"] = {
                "simple_mean": float(r.mean()),
                "value_weighted": float(lab[f"W_{a}"].sum() / lab[f"W_{b}"].sum()),
                "median": float(r.median()),
            }
    # 흡수 몫: net_r, net_inst, resid 를 net_f 에 회귀 (기울기 합 = -1)
    def slope(y, x):
        xc = x - x.mean()
        return float((xc * (y - y.mean())).sum() / (xc ** 2).sum())

    absorb = {}
    for src in INVESTORS:
        x = lab[f"net_{src}"]
        row = {}
        for tgt in INVESTORS:
            if tgt == src:
                continue
            row[tgt] = slope(lab[f"net_{tgt}"], x)
        row["other(resid)"] = slope(lab["resid"], x)
        row["sum_check"] = float(sum(row.values()))
        absorb[src] = row

    # 잔차(기타법인 등) 규모
    stats["resid"] = {
        "mean_abs_KRW_bn": float(lab.resid.abs().mean() / 1e9),
        "mean_KRW_bn": float(lab.resid.mean() / 1e9),
        "share_of_trading_value_pct": float((lab.resid.abs() / lab.trading_value).mean() * 100),
        "abs_ratio_to_net_foreign_pct": float((lab.resid.abs() / lab.net_foreign.abs().replace(0, np.nan)).median() * 100),
    }
    stats["contemporaneous_corr_u"] = {
        "foreign_retail": float(lab.u_foreign.corr(lab.u_retail)),
        "institution_retail": float(lab.u_institution.corr(lab.u_retail)),
        "foreign_institution": float(lab.u_foreign.corr(lab.u_institution)),
    }
    stats["W_mean_KRW_bn"] = {i: float(lab[f"W_{i}"].mean() / 1e9) for i in INVESTORS}
    return ratios, absorb, stats, lab


def induced_table(ratios, absorb):
    """외국인 -> 개인 (주 방향) + 역방향 참고. 관측 계수는 논문 표(Adam 주 사양)와 정확해."""
    observed = {
        # term: (foreign_obs_adam, foreign_obs_exact, retail_obs_adam, retail_obs_exact)
        "beta:momentum": (0.0496, 0.0502, -0.0304, -0.0383),
        "alpha:kospi_return_1d": (0.0361, 0.0360, -0.0246, -0.0238),
        "alpha:fx_level_z_252": (-0.0261, -0.0269, 0.0298, 0.0355),
        "beta:herd": (-0.0402, -0.0406, -0.0322, -0.0466),
    }
    phi_r = absorb["foreign"]["retail"]          # 외국인 1원 순매수에 대한 개인 순매수 반응(음수)
    phi_others = 1.0 + phi_r                      # 참고용
    rows = []
    for term, (f_adam, f_exact, r_adam, r_exact) in observed.items():
        for wname, wkey in (("단순평균", "simple_mean"), ("거래대금가중", "value_weighted"), ("중앙값", "median")):
            ratio = ratios["foreign/retail"][wkey]
            induced = phi_r * f_adam * ratio
            induced_exact = phi_r * f_exact * ratio
            rows.append({
                "term": term, "weighting": wname, "W_ratio_f_over_r": ratio,
                "phi_retail": phi_r,
                "observed_foreign(adam)": f_adam,
                "induced_retail(adam)": induced,
                "observed_retail(adam)": r_adam,
                "share_of_observed_pct(adam)": 100 * induced / r_adam if r_adam else np.nan,
                "observed_foreign(exact)": f_exact,
                "induced_retail(exact)": induced_exact,
                "observed_retail(exact)": r_exact,
                "share_of_observed_pct(exact)": 100 * induced_exact / r_exact if r_exact else np.nan,
            })
    return pd.DataFrame(rows), phi_r, phi_others


# ---------------------------------------------------------------- 합성 null

def fit_ar1(x: np.ndarray):
    x0, x1 = x[:-1], x[1:]
    phi = float(np.corrcoef(x0, x1)[0, 1])
    return phi


def simulate_ar1(rng, n, mean, sd, phi):
    innov_sd = sd * np.sqrt(max(1e-12, 1 - phi ** 2))
    e = rng.normal(0.0, innov_sd, size=n)
    x = np.empty(n)
    x[0] = rng.normal(0.0, sd)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return mean + x


def part2_simulation(config, raw, data, pos, label_pos):
    rng = np.random.default_rng(SEED)
    n_raw = len(raw)
    # 표적 모멘트 (973 표본의 라벨 일자 기준)
    lab = raw.iloc[label_pos]
    targets = {}
    for inv in ["foreign", "institution"]:
        u = lab[f"u_{inv}"].to_numpy()
        targets[inv] = {"mean": float(u.mean()), "sd": float(u.std(ddof=0)), "ar1": fit_ar1(u)}
    resid = lab["resid"].to_numpy()
    targets["resid"] = {"mean": float(resid.mean()), "sd": float(resid.std(ddof=0)), "ar1": fit_ar1(resid)}
    u_retail_real = lab["u_retail"].to_numpy()
    targets["retail_observed"] = {"sd": float(u_retail_real.std(ddof=0)), "ar1": fit_ar1(u_retail_real)}

    # 실측 특징 (u와 무관한 것): momentum, underwater, 컨텍스트
    features = data["features"].astype(np.float64)   # (n, 3, 3) [momentum, herd, underwater]
    contexts = data["contexts"].astype(np.float64)
    saved_feature_names = data["feature_names"].astype(str).tolist()
    fi_mom = saved_feature_names.index("momentum")
    fi_herd = saved_feature_names.index("herd")
    fi_uw = saved_feature_names.index("underwater")

    W = {inv: raw[f"W_{inv}"].to_numpy() for inv in INVESTORS}
    tv = raw["trading_value"].to_numpy()
    names3 = spec_terms("feat3", ["momentum", "herd", "underwater"], ["kospi_return_1d", "fx_level_z_252"])
    names4 = spec_terms("feat4_main", ["momentum", "herd", "underwater"], ["kospi_return_1d", "fx_level_z_252"])

    sim_rows = []
    moment_rows = []
    for s in range(N_SIM):
        # 전체 raw 구간에서 시뮬레이션 (herd의 t-1 시차 확보)
        u_f = simulate_ar1(rng, n_raw, targets["foreign"]["mean"], targets["foreign"]["sd"], targets["foreign"]["ar1"])
        u_i = simulate_ar1(rng, n_raw, targets["institution"]["mean"], targets["institution"]["sd"], targets["institution"]["ar1"])
        res = simulate_ar1(rng, n_raw, targets["resid"]["mean"], targets["resid"]["sd"], targets["resid"]["ar1"])
        net_f = u_f * W["foreign"]
        net_i = u_i * W["institution"]
        net_r = -(net_f + net_i + res)
        u_r = net_r / W["retail"]
        u_syn = {"foreign": u_f, "institution": u_i, "retail": u_r}
        u_mkt = {inv: (u_syn[inv] * W[inv]) / tv for inv in INVESTORS}  # herd용 (net/trading_value)

        if s < 50:
            lab_u_r = u_r[label_pos]
            moment_rows.append({
                "sim": s,
                "retail_sd": float(lab_u_r.std(ddof=0)),
                "retail_ar1": fit_ar1(lab_u_r),
                "corr_f_r": float(np.corrcoef(u_f[label_pos], lab_u_r)[0, 1]),
                "corr_i_r": float(np.corrcoef(u_i[label_pos], lab_u_r)[0, 1]),
                "saturation_|u_r|>1_rate": float((np.abs(lab_u_r) > 1).mean()),
            })

        # 973 표본 행으로 재구성: 상태행 t = raw pos, 라벨 = t+1
        for spec, names in (("feat3", names3), ("feat4_main", names4)):
            for inv_idx, inv in enumerate(INVESTORS):
                others = [o for o in INVESTORS if o != inv]
                herd = 0.5 * (u_mkt[others[0]][pos - 1] + u_mkt[others[1]][pos - 1])
                cols = [features[:, inv_idx, fi_mom], herd, features[:, inv_idx, fi_uw]]
                if spec == "feat4_main":
                    cols.append(u_syn[inv][pos])  # 지연 자기행동 = u_{i,t}
                X = np.column_stack(cols)
                y = u_syn[inv][label_pos]
                all_idx = np.arange(len(X))
                scaler = fit_feature_scaler(X[:, None, :], all_idx)
                scaled = transform_feature_tensor(X[:, None, :], scaler)[:, 0, :].astype(np.float64)
                cs = fit_context_scaler(contexts, all_idx)
                sc = transform_context_matrix(contexts, cs).astype(np.float64)
                design = build_design(scaled, sc, 3)
                coef = fit_exact(design, y, LAMBDA_CFG[inv])
                for name, value in zip(names, coef, strict=True):
                    sim_rows.append({"sim": s, "spec": spec, "investor": inv, "term": name, "weight": float(value)})
    sims = pd.DataFrame(sim_rows)
    sims.to_csv(OUT / "task2_null_sim_coefficients.csv", index=False)
    moments = pd.DataFrame(moment_rows)
    moments.to_csv(OUT / "task2_null_sim_moments.csv", index=False)

    q = (
        sims.groupby(["spec", "investor", "term"])["weight"]
        .agg(mean="mean", sd="std",
             q2_5=lambda x: float(np.percentile(x, 2.5)),
             q97_5=lambda x: float(np.percentile(x, 97.5)),
             abs_q95=lambda x: float(np.percentile(np.abs(x), 95)))
        .reset_index()
    )
    q.to_csv(OUT / "task2_null_sim_summary.csv", index=False)
    return targets, moments, q


def main() -> None:
    config, raw, data, pos, label_pos = load_raw_aligned()
    ratios, absorb, stats, lab = part1_arithmetic(raw, label_pos)
    table, phi_r, phi_others = induced_table(ratios, absorb)
    table.to_csv(OUT / "task2_induced_vs_observed.csv", index=False)
    with open(OUT / "task2_clearing_stats.json", "w") as f:
        json.dump({"W_ratios": ratios, "absorption_slopes": absorb, "stats": stats,
                   "phi_retail_from_foreign": phi_r}, f, indent=2, ensure_ascii=False)
    print(json.dumps({"W_ratios": ratios["foreign/retail"], "absorption": absorb["foreign"],
                      "resid": stats["resid"], "corr": stats["contemporaneous_corr_u"]},
                     indent=2, ensure_ascii=False))
    print(table.round(4).to_string(index=False))

    targets, moments, q = part2_simulation(config, raw, data, pos, label_pos)
    with open(OUT / "task2_null_sim_targets.json", "w") as f:
        json.dump(targets, f, indent=2)
    print("[sim] retail 달성 모멘트:", moments[["retail_sd", "retail_ar1", "corr_f_r", "corr_i_r"]].mean().round(3).to_dict())
    print("[done]")


if __name__ == "__main__":
    main()
