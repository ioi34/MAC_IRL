from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(output_dir: str | Path | None = None) -> logging.Logger:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(Path(output_dir) / "run.log", encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("mac_irl")

