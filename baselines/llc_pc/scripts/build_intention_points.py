#!/usr/bin/env python3
"""Fit motion-query intention points from local training futures only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


BASELINE_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from llc_pc import (  # noqa: E402
    IntentionPointKMeans,
    TensorizerConfig,
    load_config,
    load_configured_split,
    tensorize_sample,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit LLC-PC intention points from configured training labels only."
    )
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def build(config_path: str) -> Path:
    config = load_config(config_path)
    train_samples = load_configured_split(config, "train", include_future=True)
    tensor_config = TensorizerConfig.from_config(config)
    futures: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for sample in train_samples:
        tensor = tensorize_sample(sample, tensor_config)
        if tensor.future is None or tensor.future_valid_mask is None:
            raise ValueError(f"training sample has no future label: {tensor.sample_id}")
        futures.append(tensor.future)
        masks.append(tensor.future_valid_mask)
    values = np.stack(futures)
    valid = np.stack(masks)
    settings = config["intention_points"]
    estimator = IntentionPointKMeans(
        n_clusters=int(settings["n_clusters"]),
        random_state=int(settings.get("random_state", 42)),
        max_iter=int(settings.get("max_iter", 100)),
    ).fit(values, future_mask=valid, source_split="train")
    output = Path(config["paths"]["intention_points"])
    estimator.save(output)
    return output


def main() -> None:
    path = build(_arguments().config)
    print(path)


if __name__ == "__main__":
    main()
