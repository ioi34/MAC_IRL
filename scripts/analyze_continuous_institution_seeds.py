from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate institution ablations across seeds")
    parser.add_argument(
        "--analysis-root",
        default="experiments/2026-07-10/institution_seed_robustness",
    )
    parser.add_argument(
        "--runs-root",
        default="runs/continuous_institution_seed_robustness",
    )
    return parser.parse_args()


def seed_from_name(name: str) -> int:
    return int(name.removeprefix("seed"))


def load_seed_frames(root: Path, filename: str) -> pd.DataFrame:
    frames = []
    for seed_dir in sorted(root.glob("seed*")):
        path = seed_dir / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "seed", seed_from_name(seed_dir.name))
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No {filename} files found under {root}")
    return pd.concat(frames, ignore_index=True)


def aggregate_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    value_columns = [column for column in summary if column not in {"seed", "variant"}]
    mean_columns = [column for column in value_columns if not column.endswith("_std")]
    rows = []
    for variant, group in summary.groupby("variant", sort=True):
        row: dict[str, float | int | str] = {
            "variant": variant,
            "n_seeds": group["seed"].nunique(),
        }
        for column in mean_columns:
            row[column] = float(group[column].mean())
            row[f"{column}_between_seed_std"] = float(group[column].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_paired(paired: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column
        for column in paired
        if column not in {"seed", "variant", "comparator"}
    ]
    rows = []
    for (variant, comparator), group in paired.groupby(
        ["variant", "comparator"], sort=True
    ):
        row: dict[str, float | int | str] = {
            "variant": variant,
            "comparator": comparator,
            "n_seeds": group["seed"].nunique(),
            "seeds_with_lower_rmse": int((group["rmse_delta"] < 0).sum()),
        }
        for column in numeric:
            row[column] = float(group[column].mean())
            row[f"{column}_between_seed_std"] = float(group[column].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def collect_weight_summaries(runs_root: Path, filename: str) -> pd.DataFrame:
    frames = []
    for seed_dir in sorted(runs_root.glob("seed*")):
        seed = seed_from_name(seed_dir.name)
        for run_dir in sorted(seed_dir.iterdir()):
            path = run_dir / filename
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            frame.insert(0, "variant", run_dir.name)
            frame.insert(0, "seed", seed)
            frames.append(frame.loc[frame["investor"] == "institution"])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    args = parse_args()
    analysis_root = Path(args.analysis_root)
    runs_root = Path(args.runs_root)
    summary = load_seed_frames(analysis_root, "institution_ablation_summary.csv")
    paired = load_seed_frames(analysis_root, "institution_paired_comparison.csv")
    years = load_seed_frames(analysis_root, "institution_year_metrics.csv")
    aggregate_summary = aggregate_metrics(summary)
    aggregate_pair = aggregate_paired(paired)
    rewards = collect_weight_summaries(runs_root, "reward_weights_summary.csv")
    contexts = collect_weight_summaries(runs_root, "context_weights_summary.csv")

    summary.to_csv(analysis_root / "seed_run_summary.csv", index=False)
    paired.to_csv(analysis_root / "seed_paired_comparison.csv", index=False)
    years.to_csv(analysis_root / "seed_year_metrics.csv", index=False)
    aggregate_summary.to_csv(analysis_root / "seed_aggregate_summary.csv", index=False)
    aggregate_pair.to_csv(analysis_root / "seed_aggregate_paired.csv", index=False)
    rewards.to_csv(analysis_root / "seed_reward_weights.csv", index=False)
    contexts.to_csv(analysis_root / "seed_context_weights.csv", index=False)
    print(aggregate_summary.to_string(index=False))
    print("\nPaired across seeds")
    print(aggregate_pair.to_string(index=False))


if __name__ == "__main__":
    main()
