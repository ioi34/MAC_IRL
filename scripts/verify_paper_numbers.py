"""원고에 인용되는 사후 검증 수치를 재현한다.

학습 스크립트가 산출하지 않지만 논문 본문이 주장하는 세 가지를 계산한다.

1. **설계행렬의 유일성 진단** (§3.5) — 분할·유형별 ``rank(W)``와 ``cond(W)``.
   Lasso 해가 유일할 조건은 등상관집합 부분행렬 ``W_{E}``의 만계수인데,
   전체 ``W``가 열 만계수이면 임의 부분집합도 만계수이므로 검사 하나로 충분하다.
2. **표본외 R²** (§3.6) — ``cv_metrics_summary.csv``에 없는 지표.
   기준선을 분할별 학습표본 평균으로 둔 값과 0으로 둔 값을 모두 낸다.
   두 값이 다르므로 원고가 명시한 정의(학습평균)를 써야 한다.
3. **행동의 1차 자기상관** (§3.4) — persist의 예상 부호 근거.

사용 예::

    python scripts/verify_paper_numbers.py \
        --run-dir runs/continuous_reward3_persist \
        --output-dir experiments/2026-07-30/verify_paper_numbers
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EXPECTED_INDEX_KEYS = ("train_indices", "train", "train_idx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/continuous_reward3_persist")
    parser.add_argument("--dataset", default=None, help="생략하면 config_snapshot에서 읽는다")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def load_train_indices(split_dir: Path) -> np.ndarray:
    """indices.npz에서 학습 인덱스를 꺼낸다. 키 이름이 바뀌어도 견디게 한다."""
    payload = np.load(split_dir / "indices.npz")
    for key in EXPECTED_INDEX_KEYS:
        if key in payload:
            return np.asarray(payload[key]).ravel()
    raise KeyError(
        f"{split_dir/'indices.npz'}에서 학습 인덱스를 찾지 못했다. "
        f"보유 키: {list(payload.keys())}"
    )


def standardize(block: np.ndarray) -> np.ndarray:
    mean = block.mean(axis=0)
    std = block.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return (block - mean) / std


def build_design(features: np.ndarray, contexts: np.ndarray) -> np.ndarray:
    """w = [x ; rvec(x C^T) ; C] — 학습 구간에서만 표준화한다."""
    x = standardize(features)
    c = standardize(contexts)
    interaction = np.einsum("ni,nj->nij", x, c).reshape(len(x), -1)
    return np.hstack([x, interaction, c])


def uniqueness_diagnostics(
    run_dir: Path, dataset: dict, feature_index: list[int], context_index: list[int]
) -> pd.DataFrame:
    all_features = dataset["features"]
    all_contexts = dataset["contexts"]
    investors = [str(name) for name in dataset["investors"]]

    records = []
    for split_dir in sorted(run_dir.glob("split_*")):
        train = load_train_indices(split_dir)
        contexts = all_contexts[np.ix_(train, context_index)].astype(float)
        for position, investor in enumerate(investors):
            features = all_features[np.ix_(train, [position], feature_index)]
            features = features.reshape(len(train), -1).astype(float)
            design = build_design(features, contexts)
            records.append(
                {
                    "split": int(split_dir.name.split("_")[1]),
                    "investor": investor,
                    "n_train": len(train),
                    "n_columns": design.shape[1],
                    "rank": int(np.linalg.matrix_rank(design)),
                    "cond_W": float(np.linalg.cond(design)),
                    "cond_WtW": float(np.linalg.cond(design.T @ design)),
                }
            )
    return pd.DataFrame(records)


def out_of_sample_r2(run_dir: Path, actions: np.ndarray, investors: list[str]) -> pd.DataFrame:
    predictions = pd.read_csv(run_dir / "predictions.csv")

    train_mean: dict[tuple[int, str], float] = {}
    for split_dir in sorted(run_dir.glob("split_*")):
        split_id = int(split_dir.name.split("_")[1])
        train = load_train_indices(split_dir)
        for position, investor in enumerate(investors):
            train_mean[(split_id, investor)] = float(actions[train, position].mean())

    records = []
    grouped = predictions.groupby(["split", "investor"], sort=True)
    for (split_id, investor), frame in grouped:
        actual = frame["actual_action"].to_numpy(dtype=float)
        predicted = frame["predicted_action"].to_numpy(dtype=float)
        residual = float(((actual - predicted) ** 2).sum())
        baseline = train_mean[(int(split_id), str(investor))]
        records.append(
            {
                "split": int(split_id),
                "investor": str(investor),
                "n_test": len(actual),
                # 원고가 명시한 정의: 기준예측 = 분할별 학습표본 평균 행동
                "r2_train_mean": 1.0 - residual / float(((actual - baseline) ** 2).sum()),
                # 참고용: 기준예측 = 0
                "r2_zero": 1.0 - residual / float((actual**2).sum()),
            }
        )
    return pd.DataFrame(records)


def convergence_diagnostics(run_dir: Path, investors: list[str]) -> pd.DataFrame:
    """§3.5가 약속한 수렴 근거.

    유형별 epoch 수가 다르므로(외국인 75, 기관 10, 개인 20) "기관 null은 미학습 탓"이라는
    반론이 가능하다. 그 반론을 막으려면 **마지막 구간에서 더 배울 것이 남지 않았음**을
    보여야 한다. 두 축으로 본다.

    - 손실: 전체 감소분 중 앞 80% epoch가 이미 달성한 비율, 그리고 마지막 한 step의 비중.
    - 계수: 마지막 두 epoch 사이 θ 변화량의 최대 절대값과 그 상대 크기.
      우리 주장은 계수에 관한 것이므로 이쪽이 더 직접적인 근거다.
    """
    records = []
    for split_dir in sorted(run_dir.glob("split_*")):
        split_id = int(split_dir.name.split("_")[1])
        for investor in investors:
            loss = pd.read_csv(split_dir / f"{investor}_loss_history.csv")
            curve = loss["total_loss"].to_numpy(dtype=float)
            n_epochs = len(curve)
            total_drop = float(curve[0] - curve[-1])
            cut = max(1, int(np.ceil(0.8 * n_epochs))) - 1
            # 0으로 나누는 것을 피한다: 감소분이 없으면 비율은 정의하지 않는다
            drop_by_80 = float(curve[0] - curve[cut]) / total_drop if total_drop > 0 else np.nan
            final_step = (
                float(curve[-2] - curve[-1]) / total_drop
                if total_drop > 0 and n_epochs >= 2
                else np.nan
            )

            weights = pd.read_csv(split_dir / f"{investor}_weight_history.csv")
            wide = weights.pivot_table(
                index="epoch",
                columns=["parameter", "feature", "context"],
                values="weight",
                dropna=False,
            ).sort_index()
            last_two = wide.to_numpy(dtype=float)[-2:]
            step = np.abs(last_two[1] - last_two[0])
            scale = np.abs(last_two[1])
            records.append(
                {
                    "split": split_id,
                    "investor": investor,
                    "n_epochs": n_epochs,
                    "loss_first": float(curve[0]),
                    "loss_last": float(curve[-1]),
                    "drop_share_by_80pct_epochs": drop_by_80,
                    "final_step_share_of_drop": final_step,
                    "weight_final_step_max_abs": float(np.nanmax(step)),
                    "weight_final_step_rel": float(np.nanmax(step) / max(np.nanmax(scale), 1e-12)),
                }
            )
    return pd.DataFrame(records)


def action_autocorrelation(actions: np.ndarray, investors: list[str]) -> pd.DataFrame:
    records = []
    for position, investor in enumerate(investors):
        series = actions[:, position].astype(float)
        records.append(
            {
                "investor": investor,
                "n": len(series),
                "ar1": float(np.corrcoef(series[:-1], series[1:])[0, 1]),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = yaml.safe_load((run_dir / "config_snapshot.yaml").read_text())

    dataset_path = Path(args.dataset or config["paths"]["processed_dataset"])
    dataset = np.load(dataset_path, allow_pickle=False)

    saved_features = [str(name) for name in dataset["feature_names"]]
    saved_contexts = [str(name) for name in dataset["context_names"]]
    selected_features = list(config["features"]["selected"])
    selected_contexts = list(config["model"]["context_names"])
    feature_index = [saved_features.index(name) for name in selected_features]
    context_index = [saved_contexts.index(name) for name in selected_contexts]
    investors = [str(name) for name in dataset["investors"]]
    actions = dataset["actions"]

    output_dir = Path(args.output_dir or (run_dir / "paper_number_checks"))
    output_dir.mkdir(parents=True, exist_ok=True)

    uniqueness = uniqueness_diagnostics(run_dir, dataset, feature_index, context_index)
    r2 = out_of_sample_r2(run_dir, actions, investors)
    ar1 = action_autocorrelation(actions, investors)
    convergence = convergence_diagnostics(run_dir, investors)

    uniqueness.to_csv(output_dir / "design_matrix_uniqueness.csv", index=False)
    r2.to_csv(output_dir / "out_of_sample_r2_by_split.csv", index=False)
    ar1.to_csv(output_dir / "action_autocorrelation.csv", index=False)
    convergence.to_csv(output_dir / "convergence_by_split.csv", index=False)

    r2_summary = r2.groupby("investor")[["r2_train_mean", "r2_zero"]].mean().reset_index()
    r2_summary.to_csv(output_dir / "out_of_sample_r2_summary.csv", index=False)

    convergence_summary = (
        convergence.groupby("investor")
        .agg(
            n_epochs=("n_epochs", "max"),
            drop_share_by_80pct_min=("drop_share_by_80pct_epochs", "min"),
            drop_share_by_80pct_mean=("drop_share_by_80pct_epochs", "mean"),
            final_step_share_max=("final_step_share_of_drop", "max"),
            weight_step_max=("weight_final_step_max_abs", "max"),
            weight_step_rel_max=("weight_final_step_rel", "max"),
        )
        .reset_index()
    )
    convergence_summary.to_csv(output_dir / "convergence_summary.csv", index=False)

    summary = {
        "run_dir": str(run_dir),
        "dataset": str(dataset_path),
        "n_observations": int(actions.shape[0]),
        "features": selected_features,
        "contexts": selected_contexts,
        "design_columns": int(uniqueness["n_columns"].iloc[0]),
        "combinations_checked": int(len(uniqueness)),
        "rank_min": int(uniqueness["rank"].min()),
        "rank_deficient": int((uniqueness["rank"] < uniqueness["n_columns"]).sum()),
        "cond_W_max": float(uniqueness["cond_W"].max()),
        "cond_W_min": float(uniqueness["cond_W"].min()),
        "cond_WtW_max": float(uniqueness["cond_WtW"].max()),
        "worst_cond_at": uniqueness.loc[uniqueness["cond_W"].idxmax(), ["split", "investor"]].to_dict(),
        "r2_train_mean": r2_summary.set_index("investor")["r2_train_mean"].to_dict(),
        "r2_zero": r2_summary.set_index("investor")["r2_zero"].to_dict(),
        "action_ar1": ar1.set_index("investor")["ar1"].to_dict(),
        "convergence": convergence_summary.set_index("investor").to_dict(orient="index"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"[uniqueness] {summary['combinations_checked']}개 조합, 열 {summary['design_columns']}")
    print(f"  rank 최소 {summary['rank_min']}, 결손 {summary['rank_deficient']}건")
    print(f"  cond(W) 최대 {summary['cond_W_max']:.4f} (최소 {summary['cond_W_min']:.4f})")
    print(f"  최악 발생: split {summary['worst_cond_at']['split']}, {summary['worst_cond_at']['investor']}")
    print("[표본외 R^2] 기준=학습평균 / 기준=0")
    for investor in investors:
        print(
            f"  {investor:12s} {summary['r2_train_mean'][investor]:+.4f}"
            f" / {summary['r2_zero'][investor]:+.4f}"
        )
    print("[행동 AR(1)]")
    for investor in investors:
        print(f"  {investor:12s} {summary['action_ar1'][investor]:+.4f}")
    print("[수렴] epoch / 앞80%가 달성한 감소분(최소) / 마지막step 비중(최대) / θ 마지막변화(최대, 상대)")
    for investor in investors:
        row = summary["convergence"][investor]
        print(
            f"  {investor:12s} {int(row['n_epochs']):3d}"
            f"   {row['drop_share_by_80pct_min']:.4f}"
            f"   {row['final_step_share_max']:.4f}"
            f"   {row['weight_step_max']:.2e} ({row['weight_step_rel_max']:.4f})"
        )
    print(f"\n결과 저장: {output_dir}")


if __name__ == "__main__":
    main()
