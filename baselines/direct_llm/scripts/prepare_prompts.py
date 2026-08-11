#!/usr/bin/env python3
"""Prepare observation-only Direct LLM prompts from a private test MAT file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


BASELINE_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from direct_llm import (  # noqa: E402
    build_records,
    load_config,
    load_configured_split,
    prompt_config_from_config,
    write_jsonl,
)


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Direct LLM test prompts. This command loads observation fields "
            "only and never serializes ground-truth future coordinates."
        )
    )
    parser.add_argument("--config", required=True, help="Private local JSON config")
    parser.add_argument("--output", default=None, help="Override paths.test_prompts")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def prepare(args: argparse.Namespace) -> Path:
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    config = load_config(args.config)
    # Central leakage boundary: include_future remains false, and
    # adapt_observation does not access phase/time annotations either.
    samples = load_configured_split(
        config,
        "test",
        include_future=False,
        limit=args.max_samples,
    )
    observations = [sample.observation for sample in samples]
    records = build_records(
        observations,
        source_split="test",
        prompt_config=prompt_config_from_config(config),
    )
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else Path(config["paths"]["test_prompts"])
    )
    return write_jsonl(records, output, overwrite=args.overwrite)


def main() -> None:
    print(prepare(arguments()))


if __name__ == "__main__":
    main()
