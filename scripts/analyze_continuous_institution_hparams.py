from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.analyze_continuous_institution_ablation import run_metrics, summarize_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare institution continuous hyperparameters")
    parser.add_argument(
        "--baseline-dir",
        default="runs/continuous_institution_ablation/e1_persistence",
    )
    parser.add_argument(
        "--runs-root",
        default="runs/continuous_institution_hparams",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/2026-07-10/institution_hparam_analysis",
    )
    parser.add_argument(
        "--walk-forward-baseline-dir",
        default="runs/continuous_institution_walk_forward/e1_persistence",
    )
    parser.add_argument(
        "--walk-forward-root",
        default="runs/continuous_institution_walk_forward",
    )
    return parser.parse_args()


def relabel(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    result = frame.copy()
    result["variant"] = variant
    return result


def collect_weight_summary(run_dir: Path, variant: str, filename: str) -> pd.DataFrame:
    path = run_dir / filename
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame = frame.loc[frame["investor"] == "institution"].copy()
    frame.insert(0, "variant", variant)
    return frame


def paired_to_baseline(split_frame: pd.DataFrame, baseline: str) -> pd.DataFrame:
    baseline_frame = split_frame.loc[split_frame["variant"] == baseline].set_index("split")
    metric_columns = [
        column for column in split_frame.columns if column not in {"variant", "split"}
    ]
    rows = []
    for variant in sorted(split_frame["variant"].unique()):
        if variant == baseline:
            continue
        candidate = split_frame.loc[split_frame["variant"] == variant].set_index("split")
        row: dict[str, float | str] = {
            "variant": variant,
            "comparator": baseline,
            "rmse_win_rate": float((candidate["rmse"] < baseline_frame["rmse"]).mean()),
        }
        for metric in metric_columns:
            row[f"{metric}_delta"] = float(
                (candidate[metric] - baseline_frame[metric]).mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    baseline_dir = Path(args.baseline_dir)
    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = [baseline_dir]
    run_dirs.extend(
        sorted(
            path
            for path in runs_root.iterdir()
            if path.is_dir() and (path / "predictions.csv").exists()
        )
    )
    labels = ["t0_ep10_lr0p0005_l10p01", *[path.name for path in run_dirs[1:]]]

    split_frames = []
    year_frames = []
    reward_frames = []
    context_frames = []
    for run_dir, label in zip(run_dirs, labels, strict=True):
        split_frame, year_frame = run_metrics(run_dir)
        split_frames.append(relabel(split_frame, label))
        year_frames.append(relabel(year_frame, label))
        reward_frames.append(
            collect_weight_summary(run_dir, label, "reward_weights_summary.csv")
        )
        context_frames.append(
            collect_weight_summary(run_dir, label, "context_weights_summary.csv")
        )

    splits = pd.concat(split_frames, ignore_index=True)
    years = pd.concat(year_frames, ignore_index=True)
    summary = summarize_splits(splits)
    paired = paired_to_baseline(splits, labels[0])
    reward_weights = pd.concat(reward_frames, ignore_index=True)
    context_weights = pd.concat(context_frames, ignore_index=True)

    splits.to_csv(output_dir / "institution_hparam_split_metrics.csv", index=False)
    years.to_csv(output_dir / "institution_hparam_year_metrics.csv", index=False)
    summary.to_csv(output_dir / "institution_hparam_summary.csv", index=False)
    paired.to_csv(output_dir / "institution_hparam_paired.csv", index=False)
    reward_weights.to_csv(output_dir / "institution_hparam_reward_weights.csv", index=False)
    context_weights.to_csv(output_dir / "institution_hparam_context_weights.csv", index=False)

    walk_forward_dirs = [Path(args.walk_forward_baseline_dir)]
    walk_forward_dirs.extend(
        sorted(
            path
            for path in Path(args.walk_forward_root).glob("t*_hparam")
            if (path / "walk_forward_metrics.csv").exists()
        )
    )
    walk_forward_frames = []
    for run_dir in walk_forward_dirs:
        frame = pd.read_csv(run_dir / "walk_forward_metrics.csv")
        frame.insert(0, "variant", run_dir.name)
        walk_forward_frames.append(frame)
    pd.concat(walk_forward_frames, ignore_index=True).to_csv(
        output_dir / "institution_hparam_walk_forward_metrics.csv",
        index=False,
    )

    columns = [
        "variant",
        "rmse",
        "rmse_skill",
        "balanced_accuracy",
        "macro_f1",
        "pearson",
        "std_ratio",
        "tail_direction_accuracy",
    ]
    print(summary[columns].to_string(index=False))
    print("\nPaired with baseline")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
