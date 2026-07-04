from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from src.utils.config import load_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize loss history CSVs into experiments/")
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--out", default=None, help="Output CSV path (default: output_dir/loss_summary.csv)")
    return parser.parse_args()


def convergence_epoch(nlls: list[float], threshold: float = 0.001) -> int:
    final = nlls[-1]
    for i, nll in enumerate(nlls):
        if abs(nll - final) <= threshold:
            return i + 1
    return len(nlls)


def main() -> None:
    args = parse_args()
    config = load_configs(args.experiment_config)
    output_dir = Path(config["experiment"]["output_dir"])
    investors = config.get("investors", ["foreign", "institution", "retail"])

    rows = []
    trajectory_frames = []
    split_dirs = sorted(output_dir.glob("split_*"))
    if not split_dirs:
        print(f"No split directories found in {output_dir}")
        return

    for split_dir in split_dirs:
        split_id = split_dir.name
        for investor in investors:
            history_path = split_dir / f"{investor}_loss_history.csv"
            if not history_path.exists():
                continue
            with history_path.open() as f:
                reader = csv.DictReader(f)
                history = list(reader)
            if not history:
                continue
            trajectory = pd.DataFrame(history).astype(
                {
                    "epoch": int,
                    "train_nll": float,
                    "l1_penalty": float,
                    "total_loss": float,
                }
            )
            trajectory.insert(0, "investor", investor)
            trajectory.insert(0, "split", split_id)
            trajectory_frames.append(trajectory)
            nlls = [float(r["train_nll"]) for r in history]
            rows.append({
                "split": split_id,
                "investor": investor,
                "initial_nll": round(nlls[0], 6),
                "final_nll": round(nlls[-1], 6),
                "nll_drop": round(nlls[0] - nlls[-1], 6),
                "convergence_epoch": convergence_epoch(nlls),
                "total_epochs": len(nlls),
            })

    out_path = Path(args.out) if args.out else output_dir / "loss_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "investor", "initial_nll", "final_nll", "nll_drop", "convergence_epoch", "total_epochs"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    selected_epochs = [1, 5, 10, 20, 50, 100, 200]
    trajectory = pd.concat(trajectory_frames, ignore_index=True)
    trajectory = trajectory[trajectory["epoch"].isin(selected_epochs)]
    trajectory_summary = (
        trajectory.groupby(["investor", "epoch"], sort=False)[
            ["train_nll", "l1_penalty", "total_loss"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    trajectory_summary.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in trajectory_summary.columns
    ]
    trajectory_summary.to_csv(output_dir / "loss_trajectory_summary.csv", index=False)

    # 투자자별 평균 요약 출력
    print(f"\nLoss summary → {out_path}\n")
    print(f"{'investor':<12} {'mean_final_nll':>15} {'mean_convergence_epoch':>22} {'mean_nll_drop':>15}")
    print("-" * 68)
    for investor in investors:
        inv_rows = [r for r in rows if r["investor"] == investor]
        if not inv_rows:
            continue
        mean_final = sum(r["final_nll"] for r in inv_rows) / len(inv_rows)
        mean_conv = sum(r["convergence_epoch"] for r in inv_rows) / len(inv_rows)
        mean_drop = sum(r["nll_drop"] for r in inv_rows) / len(inv_rows)
        print(f"{investor:<12} {mean_final:>15.6f} {mean_conv:>22.1f} {mean_drop:>15.6f}")


if __name__ == "__main__":
    main()
