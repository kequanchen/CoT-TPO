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
            "Build adapted LC-LLM JSONL. Supervised mode is restricted to the "
            "training split; inference mode never loads y_future."
        )
    )
    parser.add_argument("--config", required=True, help="Path to a private local JSON config")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument(
        "--mode",
        choices=("supervised", "inference"),
        default=None,
        help="Default: supervised for train and inference for test",
    )
    parser.add_argument("--output", default=None, help="Override the configured JSONL path")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare(args: argparse.Namespace) -> Path:
    mode = args.mode or ("supervised" if args.split == "train" else "inference")
    supervised = mode == "supervised"
    if supervised and args.split != "train":
        raise ValueError("supervised dataset generation is allowed only for --split train")
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
        path_key = "train_jsonl" if supervised else f"{args.split}_jsonl"
        configured = config["paths"].get(path_key)
        if configured is None:
            configured = Path(config["paths"]["adapter_dir"]) / f"{args.split}_prompts.jsonl"
        output = Path(configured)
    return write_jsonl(records, output, overwrite=args.overwrite)


def main() -> None:
    output = prepare(arguments())
    print(output)


if __name__ == "__main__":
    main()
