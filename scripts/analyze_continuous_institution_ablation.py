from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare continuous institution ablations")
    parser.add_argument(
        "--runs-root",
        default="runs/continuous_institution_ablation",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/2026-07-10/institution_current_data_analysis",
    )
    return parser.parse_args()


def safe_correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    if np.std(actual) == 0 or np.std(predicted) == 0:
        return np.nan
    return float(np.corrcoef(actual, predicted)[0, 1])


def split_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, float]:
    error = predicted - actual
    baseline_error = baseline - actual
    rmse = float(np.sqrt(np.mean(np.square(error))))
    baseline_rmse = float(np.sqrt(np.mean(np.square(baseline_error))))
    mae = float(np.mean(np.abs(error)))
    baseline_mae = float(np.mean(np.abs(baseline_error)))
    actual_direction = actual >= 0
    predicted_direction = predicted >= 0
    tail_cutoff = float(np.quantile(np.abs(actual), 0.75))
    tail = np.abs(actual) >= tail_cutoff
    return {
        "rmse": rmse,
        "nrmse": rmse / float(np.std(actual)),
        "rmse_skill": 1.0 - rmse / baseline_rmse,
        "mse_skill": 1.0 - np.mean(np.square(error)) / np.mean(np.square(baseline_error)),
        "mae_skill": 1.0 - mae / baseline_mae,
        "direction_accuracy": float(np.mean(actual_direction == predicted_direction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(actual_direction, predicted_direction)
        ),
        "macro_f1": float(
            f1_score(
                actual_direction,
                predicted_direction,
                average="macro",
                labels=[False, True],
                zero_division=0,
            )
        ),
        "pearson": safe_correlation(actual, predicted),
        "spearman": float(pd.Series(actual).corr(pd.Series(predicted), method="spearman")),
        "std_ratio": float(np.std(predicted) / np.std(actual)),
        "tail_direction_accuracy": float(
            np.mean(actual_direction[tail] == predicted_direction[tail])
        ),
    }


def run_metrics(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_yaml(run_dir / "config_snapshot.yaml")
    data = np.load(config["paths"]["processed_dataset"], allow_pickle=False)
    investors = data["investors"].astype(str).tolist()
    institution_idx = investors.index("institution")
    actions = data["actions"][:, institution_idx]
    predictions = pd.read_csv(run_dir / "predictions.csv")
    predictions = predictions.loc[predictions["investor"] == "institution"].copy()
    variant = run_dir.name

    rows = []
    for split, group in predictions.groupby("split", sort=True):
        indices = np.load(run_dir / f"split_{int(split):02d}" / "indices.npz")
        train_indices = indices["train_indices"]
        observation_indices = group["observation_index"].to_numpy(dtype=int)
        actual = actions[observation_indices]
        predicted = group["predicted_action"].to_numpy(dtype=float)
        baseline = np.full_like(actual, actions[train_indices].mean())
        rows.append(
            {
                "variant": variant,
                "split": int(split),
                **split_metrics(actual, predicted, baseline),
            }
        )

    averaged = (
        predictions.groupby(["observation_index", "date"], as_index=False)
        .agg(actual_action=("actual_action", "first"), predicted_action=("predicted_action", "mean"))
    )
    averaged["year"] = averaged["date"].astype(str).str[:4]
    year_rows = []
    for year, group in averaged.groupby("year", sort=True):
        actual = group["actual_action"].to_numpy(dtype=float)
        predicted = group["predicted_action"].to_numpy(dtype=float)
        actual_direction = actual >= 0
        predicted_direction = predicted >= 0
        year_rows.append(
            {
                "variant": variant,
                "year": int(year),
                "n": len(group),
                "rmse": float(np.sqrt(np.mean(np.square(predicted - actual)))),
                "direction_accuracy": float(
                    np.mean(actual_direction == predicted_direction)
                ),
                "balanced_accuracy": float(
                    balanced_accuracy_score(actual_direction, predicted_direction)
                ),
                "macro_f1": float(
                    f1_score(
                        actual_direction,
                        predicted_direction,
                        average="macro",
                        labels=[False, True],
                        zero_division=0,
                    )
                ),
                "pearson": safe_correlation(actual, predicted),
                "std_ratio": float(np.std(predicted) / np.std(actual)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(year_rows)


def summarize_splits(split_frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [column for column in split_frame if column not in {"variant", "split"}]
    rows = []
    for variant, group in split_frame.groupby("variant", sort=True):
        row: dict[str, float | str] = {"variant": variant}
        for metric in metrics:
            row[metric] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_comparison(split_frame: pd.DataFrame) -> pd.DataFrame:
    variants = sorted(split_frame["variant"].unique())
    baseline = next(variant for variant in variants if variant.startswith("e0_"))
    persistence = next(
        (variant for variant in variants if variant.startswith("e1_")),
        None,
    )
    drift = next(
        (variant for variant in variants if variant.startswith("e4_")),
        None,
    )
    rows = []
    for variant in variants:
        if variant == baseline:
            continue
        if variant.startswith(("e6_", "e7_")):
            if persistence is None:
                raise ValueError(f"{variant} requires an e1 persistence comparator")
            comparator = persistence
        elif variant.startswith("e5_"):
            if drift is None:
                raise ValueError(f"{variant} requires an e4 drift comparator")
            comparator = drift
        else:
            comparator = baseline
        candidate = split_frame.loc[split_frame["variant"] == variant].set_index("split")
        reference = split_frame.loc[split_frame["variant"] == comparator].set_index("split")
        rows.append(
            {
                "variant": variant,
                "comparator": comparator,
                "rmse_delta": float((candidate["rmse"] - reference["rmse"]).mean()),
                "rmse_skill_delta": float(
                    (candidate["rmse_skill"] - reference["rmse_skill"]).mean()
                ),
                "rmse_win_rate": float((candidate["rmse"] < reference["rmse"]).mean()),
                "balanced_accuracy_delta": float(
                    (
                        candidate["balanced_accuracy"]
                        - reference["balanced_accuracy"]
                    ).mean()
                ),
                "pearson_delta": float(
                    (candidate["pearson"] - reference["pearson"]).mean()
                ),
                "std_ratio_delta": float(
                    (candidate["std_ratio"] - reference["std_ratio"]).mean()
                ),
                "tail_direction_delta": float(
                    (
                        candidate["tail_direction_accuracy"]
                        - reference["tail_direction_accuracy"]
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def collect_weights(run_dirs: list[Path], filename: str) -> pd.DataFrame:
    frames = []
    for run_dir in run_dirs:
        path = run_dir / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "variant", run_dir.name)
        frames.append(frame.loc[frame["investor"] == "institution"])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(
        path
        for path in runs_root.iterdir()
        if path.is_dir() and (path / "predictions.csv").exists()
    )
    if not run_dirs:
        raise FileNotFoundError(f"No completed runs found under {runs_root}")

    split_frames = []
    year_frames = []
    for run_dir in run_dirs:
        split_frame, year_frame = run_metrics(run_dir)
        split_frames.append(split_frame)
        year_frames.append(year_frame)
    splits = pd.concat(split_frames, ignore_index=True)
    years = pd.concat(year_frames, ignore_index=True)
    summary = summarize_splits(splits)
    paired = paired_comparison(splits)
    reward_weights = collect_weights(run_dirs, "reward_weights_summary.csv")
    context_weights = collect_weights(run_dirs, "context_weights_summary.csv")

    splits.to_csv(output_dir / "institution_split_metrics.csv", index=False)
    years.to_csv(output_dir / "institution_year_metrics.csv", index=False)
    summary.to_csv(output_dir / "institution_ablation_summary.csv", index=False)
    paired.to_csv(output_dir / "institution_paired_comparison.csv", index=False)
    reward_weights.to_csv(output_dir / "institution_reward_weights.csv", index=False)
    context_weights.to_csv(output_dir / "institution_context_weights.csv", index=False)
    print(summary.to_string(index=False))
    print("\nPaired comparisons")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
