#!/usr/bin/env python3
"""Build the training-only LLC-PC semantic-context retrieval index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


BASELINE_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from llc_pc import (  # noqa: E402
    CONTEXT_DIM,
    FeatureStandardizer,
    TensorizerConfig,
    TrainContextIndex,
    load_config,
    load_configured_split,
    standardizer_path_for_index,
    tensorize_samples,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an LLC-PC index from training contexts only.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--contexts",
        default=None,
        help="Optional training contexts JSONL; defaults to raw_context_dir/train/contexts.jsonl",
    )
    return parser.parse_args()


def _read_contexts(path: Path) -> dict[str, tuple[np.ndarray, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"training context file not found: {path}")
    records: dict[str, tuple[np.ndarray, str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if str(record.get("source_split", "")).lower() != "train":
                raise ValueError(f"line {line_number} is not marked source_split=train")
            if record.get("status") != "ok":
                continue
            sample_id = str(record.get("sample_id", ""))
            if not sample_id or sample_id in records:
                raise ValueError(f"line {line_number} has an empty or duplicate sample_id")
            vector = np.asarray(record.get("context_vector"), dtype=np.float32)
            if vector.shape != (CONTEXT_DIM,) or not np.isfinite(vector).all():
                raise ValueError(f"line {line_number} has an invalid 17-D context_vector")
            records[sample_id] = (vector, str(record.get("event_id", "")))
    if not records:
        raise ValueError("no validated training contexts were found")
    return records


def build(config_path: str, contexts_path: str | None = None) -> tuple[Path, Path]:
    config = load_config(config_path)
    source = (
        Path(contexts_path).expanduser().resolve()
        if contexts_path
        else Path(config["paths"]["raw_context_dir"]) / "train" / "contexts.jsonl"
    )
    context_records = _read_contexts(source)

    # No labels are loaded: retrieval features are observation-only by construction.
    train_samples = load_configured_split(config, "train", include_future=False)
    batch = tensorize_samples(train_samples, TensorizerConfig.from_config(config))
    standardizer = FeatureStandardizer()
    standardized = standardizer.fit_transform(batch.retrieval_features, source_split="train")
    row_by_id = {sample_id: index for index, sample_id in enumerate(batch.sample_ids.tolist())}
    if len(row_by_id) != len(batch.sample_ids):
        raise ValueError("training data contain duplicate sample IDs")

    selected_rows: list[int] = []
    contexts: list[np.ndarray] = []
    sample_ids: list[str] = []
    event_ids: list[str] = []
    for sample_id, (vector, record_event) in context_records.items():
        if sample_id not in row_by_id:
            raise ValueError(f"context sample is absent from configured training data: {sample_id}")
        row = row_by_id[sample_id]
        data_event = str(batch.event_ids[row])
        if record_event and record_event != data_event:
            raise ValueError(f"event ID mismatch for training context {sample_id}")
        selected_rows.append(row)
        contexts.append(vector)
        sample_ids.append(sample_id)
        event_ids.append(data_event)

    context_cfg = config["context"]
    index = TrainContextIndex(
        metric=str(context_cfg.get("metric", "euclidean")),
        context_dim=int(context_cfg.get("dimension", CONTEXT_DIM)),
    ).fit(
        standardized[np.asarray(selected_rows, dtype=np.int64)],
        np.stack(contexts),
        sample_ids,
        event_ids=event_ids,
        source_split="train",
    )
    index_path = Path(config["paths"]["context_index"])
    standardizer_path = standardizer_path_for_index(index_path)
    index.save(index_path)
    standardizer.save(standardizer_path)
    return index_path, standardizer_path


def main() -> None:
    args = _arguments()
    index, standardizer = build(args.config, args.contexts)
    print(index)
    print(standardizer)


if __name__ == "__main__":
    main()
