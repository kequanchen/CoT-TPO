#!/usr/bin/env python3
"""Prepare private LC-LLM SFT or prompt-only JSONL records locally."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


BASELINE_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from lc_llm import (  # noqa: E402
    build_records,
    label_config_from_config,
    load_config,
    load_configured_split,
    prompt_config_from_config,
    road_config_from_config,
    write_jsonl,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build adapted LC-LLM JSONL. Supervised mode is restricted to train "
            "and validation; inference mode never loads y_future."
        )
    )
    parser.add_argument("--config", required=True, help="Path to a private local JSON config")
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="train"
    )
    parser.add_argument(
        "--mode",
        choices=("supervised", "inference"),
        default=None,
        help="Default: supervised for train/validation and inference for test",
    )
    parser.add_argument("--output", default=None, help="Override the configured JSONL path")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare(args: argparse.Namespace) -> Path:
    mode = _resolve_mode(args.split, args.mode)
    supervised = mode == "supervised"
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    config = load_config(args.config)
    # Central leakage boundary: prompt-only inference does not load y_future.
    samples = load_configured_split(
        config,
        args.split,
        include_future=supervised,
        limit=args.max_samples,
    )
    records = build_records(
        samples,
        source_split=args.split,
        supervised=supervised,
        road_config=road_config_from_config(config),
        prompt_config=prompt_config_from_config(config),
        label_config=label_config_from_config(config),
    )
    if args.output:
        output = Path(args.output).expanduser().resolve()
    else:
        path_key = f"{args.split}_jsonl"
        configured = config["paths"].get(path_key)
        if configured is None:
            configured = Path(config["paths"]["adapter_dir"]) / f"{args.split}_prompts.jsonl"
        output = Path(configured)
    return write_jsonl(records, output, overwrite=args.overwrite)


def _resolve_mode(split: str, mode: str | None) -> str:
    normalized = str(split).strip().lower()
    if normalized not in {"train", "validation", "test"}:
        raise ValueError("split must be 'train', 'validation', or 'test'")
    resolved = mode or ("inference" if normalized == "test" else "supervised")
    if resolved == "supervised" and normalized == "test":
        raise ValueError("supervised dataset generation is forbidden for --split test")
    if resolved not in {"supervised", "inference"}:
        raise ValueError("mode must be 'supervised' or 'inference'")
    return resolved


def main() -> None:
    output = prepare(arguments())
    print(output)


if __name__ == "__main__":
    main()
