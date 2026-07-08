from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.contexts import build_context_matrix
from src.data.continuous_labels import add_continuous_action_labels, continuous_actions_array
from src.data.loaders import load_daily_frame
from src.data.preprocess import prepare_daily_frame
from src.features.continuous import build_state_feature_tensor, valid_rows_for_continuous_model
from src.utils.config import load_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare continuous-action MAC-IRL features")
    parser.add_argument("--data-config", default="configs/data_continuous.yaml")
    parser.add_argument("--features-config", default="configs/features_continuous.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_configs(args.data_config, args.features_config)
    raw = load_daily_frame(config)
    daily = prepare_daily_frame(raw, config)
    labeled = add_continuous_action_labels(daily, config)
    features, feature_names = build_state_feature_tensor(labeled, config)
    contexts = None
    context_names = []
    if config.get("contexts", {}).get("selected"):
        contexts, context_names = build_context_matrix(labeled, config)

    date_col = config["columns"]["date"]
    dates = pd.to_datetime(labeled[date_col])
    sample = config["sample"]
    if sample.get("selection", "latest") != "latest":
        raise ValueError("Only sample.selection='latest' is supported")
    in_period = (dates >= pd.Timestamp(sample["start"])) & (dates <= pd.Timestamp(sample["end"]))
    valid = valid_rows_for_continuous_model(labeled, features, config["investors"], contexts)
    eligible_indices = np.flatnonzero(in_period.to_numpy() & valid)
    sample_size = int(sample["size"])
    if len(eligible_indices) < sample_size:
        raise ValueError(
            f"Only {len(eligible_indices)} valid observations are available in "
            f"{sample['start']}..{sample['end']}; {sample_size} are required"
        )
    selected_indices = eligible_indices[-sample_size:]
    selected = labeled.iloc[selected_indices].reset_index(drop=True)
    selected_features = features[selected_indices]
    selected_contexts = contexts[selected_indices] if contexts is not None else None
    actions = continuous_actions_array(selected, config["investors"])

    output_path = Path(config["paths"]["processed_dataset"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "features": selected_features.astype(np.float32),
        "actions": actions.astype(np.float32),
        "dates": np.asarray(selected[date_col].dt.strftime("%Y-%m-%d"), dtype="U10"),
        "feature_names": np.asarray(feature_names, dtype=str),
        "investors": np.asarray(config["investors"], dtype=str),
    }
    if selected_contexts is not None:
        arrays["contexts"] = selected_contexts.astype(np.float32)
        arrays["context_names"] = np.asarray(context_names, dtype=str)
    np.savez_compressed(output_path, **arrays)

    metadata = {
        "num_rows": len(selected),
        "date_start": str(selected[date_col].iloc[0].date()),
        "date_end": str(selected[date_col].iloc[-1].date()),
        "investors": config["investors"],
        "feature_names": feature_names,
        "feature_shape": list(selected_features.shape),
        "source_rows": len(raw),
        "eligible_rows": len(eligible_indices),
        "rows_excluded_before_sample": int(selected_indices[0]),
        "label_definition": "(buy_value - sell_value) / (buy_value + sell_value), shifted by label_shift",
        "policy_definition": "clip((beta + B C_t)^T x_t, -1, 1)",
        "scaling": "deferred_to_each_cpcv_training_split",
        "action_summary": {
            investor: {
                "mean": float(actions[:, investor_idx].mean()),
                "std": float(actions[:, investor_idx].std(ddof=0)),
                "min": float(actions[:, investor_idx].min()),
                "max": float(actions[:, investor_idx].max()),
                "positive_rate": float((actions[:, investor_idx] > 0).mean()),
                "negative_rate": float((actions[:, investor_idx] < 0).mean()),
            }
            for investor_idx, investor in enumerate(config["investors"])
        },
    }
    if selected_contexts is not None:
        metadata["context_names"] = context_names
        metadata["context_shape"] = list(selected_contexts.shape)
    metadata_path = Path(config["paths"]["processed_metadata"])
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
