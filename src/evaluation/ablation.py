from __future__ import annotations

from copy import deepcopy


def remove_features(config: dict, features_to_remove: set[str]) -> dict:
    updated = deepcopy(config)
    updated["features"]["selected"] = [
        name for name in updated["features"]["selected"] if name not in features_to_remove
    ]
    return updated

