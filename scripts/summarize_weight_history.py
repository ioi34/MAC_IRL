from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.config import load_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize epoch-level reward weight histories across CPCV splits"
    )
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument(
        "--epochs",
        nargs="+",
        type=int,
        default=[0, 1, 5, 10, 20, 50, 100, 200],
    )
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_configs(args.experiment_config)
    output_dir = Path(config["experiment"]["output_dir"])
    frames = []
    for split_dir in sorted(output_dir.glob("split_*")):
        split = int(split_dir.name.removeprefix("split_"))
        for history_path in sorted(split_dir.glob("*_weight_history.csv")):
            frame = pd.read_csv(history_path, keep_default_na=False)
            frame.insert(0, "investor", history_path.stem.removesuffix("_weight_history"))
            frame.insert(0, "split", split)
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No weight history CSVs found under {output_dir}")

    history = pd.concat(frames, ignore_index=True)
    history = history[history["epoch"].isin(args.epochs)]
    group_columns = ["epoch", "investor", "parameter", "feature", "context"]
    summary = (
        history.groupby(group_columns, sort=False, dropna=False)["weight"]
        .agg(["mean", "std"])
        .reset_index()
    )
    signs = history.assign(
        positive=history["weight"] > 0,
        negative=history["weight"] < 0,
    )
    sign_summary = (
        signs.groupby(group_columns, sort=False, dropna=False)[["positive", "negative"]]
        .mean()
        .rename(columns={"positive": "positive_rate", "negative": "negative_rate"})
        .reset_index()
    )
    summary = summary.merge(sign_summary, on=group_columns)
    summary["direction_consistency"] = summary[
        ["positive_rate", "negative_rate"]
    ].max(axis=1)

    out_path = (
        Path(args.out)
        if args.out
        else output_dir / "weight_trajectory_summary.csv"
    )
    summary.to_csv(out_path, index=False)
    final_epoch = max(args.epochs)
    for parameter, filename in [
        ("beta", "beta_trajectory_table.csv"),
        ("B", "context_weight_trajectory_table.csv"),
    ]:
        parameter_summary = summary[summary["parameter"] == parameter]
        if parameter_summary.empty:
            continue
        index_columns = ["investor", "feature"]
        if parameter == "B":
            index_columns.append("context")
        means = parameter_summary.pivot(
            index=index_columns,
            columns="epoch",
            values="mean",
        )
        means.columns = [f"epoch_{epoch}_mean" for epoch in means.columns]
        means = means.reset_index()
        final = parameter_summary[parameter_summary["epoch"] == final_epoch][
            index_columns + ["std", "direction_consistency"]
        ].rename(
            columns={
                "std": f"epoch_{final_epoch}_std",
                "direction_consistency": f"epoch_{final_epoch}_direction_consistency",
            }
        )
        means.merge(final, on=index_columns).to_csv(
            output_dir / filename,
            index=False,
        )
    print(f"Weight trajectory summary → {out_path}")


if __name__ == "__main__":
    main()
