from __future__ import annotations

import pandas as pd


def reward_weights_frame(
    model,
    investor: str,
    feature_names: list[str],
    split: int,
) -> pd.DataFrame:
    beta = model.reward_model.beta.detach().cpu().numpy()
    return pd.DataFrame(
        {
            "split": split,
            "investor": investor,
            "feature": feature_names,
            "weight": beta.astype(float),
        }
    )


def summarize_reward_weights(weights: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (investor, feature), group in weights.groupby(["investor", "feature"], sort=False):
        values = group["weight"]
        mean = float(values.mean())
        dominant_sign = 1 if mean > 0 else -1 if mean < 0 else 0
        signs = values.map(lambda value: 1 if value > 0 else -1 if value < 0 else 0)
        rows.append(
            {
                "investor": investor,
                "feature": feature,
                "mean": mean,
                "std": float(values.std(ddof=1)),
                "positive_rate": float((signs > 0).mean()),
                "negative_rate": float((signs < 0).mean()),
                "dominant_direction": {1: "positive", -1: "negative", 0: "zero"}[dominant_sign],
                "direction_consistency": float((signs == dominant_sign).mean()),
            }
        )
    return pd.DataFrame(rows)
