"""Adam 적합 계수를 정확 Lasso(coordinate descent) 해와 대조한다.

원고 §3.5는 목적함수가 볼록이고 유일해를 가짐을 rank/등상관집합으로 검증해놓고
정작 추정은 Adam으로 한다. 리뷰어 지적(단점 2)이 여기다. 이 스크립트는 동일한 45개
CPCV split·3유형에 대해 정확 솔버로 다시 풀어 두 해가 일치함을 보인다.

목적함수 대응:
    원고    (1/n) Σ (θᵀw - y)² + λ‖θ‖₁            , λ = 0.005
    sklearn (1/2n)‖y - Xθ‖²    + α‖θ‖₁
  원고 = 2 × sklearn  이므로  α = λ/2 = 0.0025.

설계행렬은 src/models/continuous.py 의 forward 를 그대로 옮긴 것:
    state_score = Σ_f x_f β_f + Σ_f x_f (Σ_c c_c B_fc) + Σ_c c_c α_c
    => w = [ x (3) ; rvec(x cᵀ) (6) ; c (2) ]   총 11

사용:
  python3 scripts/compare_exact_lasso.py -o experiments/2026-07-31/exact_lasso
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

RUN = Path("runs/continuous_reward3_persist_epochs75")
DATA = Path("data/processed/dataset_continuous_reward3_persist.npz")
LAMBDA = 0.005
INVESTORS = ["foreign", "institution", "retail"]


def resolve_data_path(run: Path) -> Path:
    """run/config_snapshot.yaml의 paths.processed_dataset을 따라간다.

    run-dir이 바뀌면 그 run이 실제로 쓴 데이터셋(예: persist_lag0)도 같이
    바뀌어야 split indices.npz의 행 번호가 어긋나지 않는다. 스냅샷을 못 읽으면
    기존 하드코딩 DATA로 폴백한다.
    """
    snapshot = run / "config_snapshot.yaml"
    if not snapshot.exists():
        return DATA
    with snapshot.open() as f:
        config = yaml.safe_load(f)
    path = config.get("paths", {}).get("processed_dataset")
    return Path(path) if path else DATA


def design(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    """[x ; rvec(x cᵀ) ; c] — forward 의 항 순서와 동일."""
    inter = (x[:, :, None] * c[:, None, :]).reshape(len(x), -1)  # row-major: feature-major
    return np.hstack([x, inter, c])


def adam_theta(split: int, investor: str, features: list[str], contexts: list[str],
               beta_df, ctx_df, main_df) -> np.ndarray:
    b = beta_df[(beta_df.split == split) & (beta_df.investor == investor)].set_index("feature")["weight"]
    B = ctx_df[(ctx_df.split == split) & (ctx_df.investor == investor)].set_index(["feature", "context"])["weight"]
    a = main_df[(main_df.split == split) & (main_df.investor == investor)].set_index("context")["weight"]
    return np.concatenate([
        [b[f] for f in features],
        [B[(f, c)] for f in features for c in contexts],
        [a[c] for c in contexts],
    ])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--out-dir", default="experiments/2026-07-31/exact_lasso")
    p.add_argument("--run-dir", type=Path, default=RUN)
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run = args.run_dir
    data_path = resolve_data_path(run)

    d = np.load(data_path, allow_pickle=False)
    feature_names = d["feature_names"].astype(str).tolist()
    context_names = d["context_names"].astype(str).tolist()
    features, actions, contexts = d["features"], d["actions"], d["contexts"]

    beta_df = pd.read_csv(run / "reward_weights.csv")
    ctx_df = pd.read_csv(run / "context_weights.csv")
    main_df = pd.read_csv(run / "context_main_weights.csv")

    names = (feature_names
             + [f"{f}x{c}" for f in feature_names for c in context_names]
             + context_names)

    rows, sat_rows = [], []
    for split_dir in sorted(run.glob("split_*")):
        split = int(split_dir.name.split("_")[1])
        idx = np.load(split_dir / "indices.npz")["train_indices"]
        cs = StandardScaler().fit(contexts[idx])
        c_all = cs.transform(contexts).astype(np.float32)

        for i, investor in enumerate(INVESTORS):
            fx = features[:, i]
            fs = StandardScaler().fit(fx[idx])
            x_all = fs.transform(fx).astype(np.float32)

            W = design(x_all[idx], c_all[idx]).astype(np.float64)
            y = actions[idx, i].astype(np.float64)

            fit = Lasso(alpha=LAMBDA / 2, fit_intercept=False, max_iter=200_000, tol=1e-12).fit(W, y)
            exact = fit.coef_
            adam = adam_theta(split, investor, feature_names, context_names, beta_df, ctx_df, main_df)

            # clip 이 실제로 결속되는지 (원고: 포화율 0)
            sat = float(np.mean(np.abs(W @ exact) > 1.0))
            sat_rows.append({"split": split, "investor": investor, "saturation_rate": sat})

            for name, e, a in zip(names, exact, adam):
                rows.append({"split": split, "investor": investor, "coefficient": name,
                             "exact_lasso": e, "adam": a, "abs_diff": abs(e - a)})

    df = pd.DataFrame(rows)
    df.to_csv(out / "coefficient_comparison.csv", index=False)
    pd.DataFrame(sat_rows).to_csv(out / "saturation.csv", index=False)

    # 요약 1: 유형별 최대/평균 차이, 부호 불일치 건수
    summary = (df.assign(sign_mismatch=lambda t: np.sign(t.exact_lasso) != np.sign(t.adam))
                 .groupby("investor")
                 .agg(max_abs_diff=("abs_diff", "max"),
                      mean_abs_diff=("abs_diff", "mean"),
                      n=("abs_diff", "size"),
                      sign_mismatches=("sign_mismatch", "sum")))
    summary.to_csv(out / "summary_by_investor.csv")

    # 요약 2: 해석 대상 5계수(beta 3 + alpha 2)의 CPCV 평균 — 원고 Table 2/3 대조용
    interp = feature_names + context_names
    means = (df[df.coefficient.isin(interp)]
             .groupby(["investor", "coefficient"])[["exact_lasso", "adam"]]
             .agg(["mean", "std"]))
    means.columns = ["_".join(c) for c in means.columns]
    means["mean_diff"] = means["exact_lasso_mean"] - means["adam_mean"]
    means.to_csv(out / "interpreted_coefficients.csv")

    print(summary.to_string())
    print()
    print(means.round(5).to_string())
    print(f"\n최대 포화율: {pd.DataFrame(sat_rows).saturation_rate.max():.6f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
