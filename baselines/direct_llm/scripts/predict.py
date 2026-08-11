#!/usr/bin/env python3
"""Run the adapted Direct LLM baseline through an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


BASELINE_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate restartable Direct LLM trajectory predictions from prepared prompts."
    )
    parser.add_argument("--config", required=True, help="Path to a Direct LLM JSON configuration")
    parser.add_argument("--input", default=None, help="Override paths.test_prompts")
    parser.add_argument("--output", default=None, help="Override paths.predictions")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip sample IDs already present in output (default: true)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-resumed output")
    return parser.parse_args()


def predict(args: argparse.Namespace) -> tuple[Path, dict[str, int]]:
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot both be enabled")
    from direct_llm.config import load_config

    config = load_config(Path(args.config).expanduser().resolve())
    paths = _section(config, "paths")
    llm = _section(config, "llm")
    data = _section(config, "data")
    _reject_inline_credentials(llm)

    from direct_llm.client import (
        complete_with_retries,
        create_openai_client,
        settings_from_mapping,
    )
    from direct_llm.generation import load_prompt_records, run_predictions
    from direct_llm.schema import parse_direct_output

    input_path = _configured_path(
        args.input or paths.get("test_prompts"),
        "paths.test_prompts",
    )
    output_path = _configured_path(
        args.output or paths.get("predictions"),
        "paths.predictions",
        create_parent=True,
    )
    expected_points = int(data.get("future_steps", 50))
    if expected_points <= 0:
        raise ValueError("data.future_steps must be positive")
    max_format_retries = int(llm.get("max_format_retries", 2))
    if max_format_retries < 0:
        raise ValueError("llm.max_format_retries cannot be negative")

    settings = settings_from_mapping(llm)
    records = load_prompt_records(input_path, max_records=args.max_samples)
    client = create_openai_client(settings)

    def completion(record, reminder):
        return complete_with_retries(
            client,
            settings,
            system_prompt=record.system_prompt,
            user_prompt=record.user_prompt,
            format_reminder=reminder,
        )

    def strict_parser(text: str):
        return parse_direct_output(text, expected_points=expected_points)

    counts = run_predictions(
        records,
        output_path,
        completion,
        strict_parser,
        expected_points=expected_points,
        max_format_retries=max_format_retries,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    return output_path, counts


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = config.get(name)
    if not isinstance(section, Mapping):
        raise ValueError(f"missing configuration object: {name}")
    return section


def _reject_inline_credentials(llm: Mapping[str, Any]) -> None:
    forbidden = sorted(
        key
        for key in llm
        if str(key).casefold()
        in {"api_key", "apikey", "base_url", "token", "access_token", "secret"}
    )
    if forbidden:
        raise ValueError(
            "credentials and endpoint URLs must be supplied only through the environment "
            "variables named by api_key_env/base_url_env; remove: "
            + ", ".join(str(item) for item in forbidden)
        )


def _configured_path(value: Any, field: str, *, create_parent: bool = False) -> Path:
    if not isinstance(value, str) or not value.strip() or _placeholder(value):
        raise ValueError(f"{field} must be set in a private config or CLI argument")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = BASELINE_ROOT / path
    path = path.resolve()
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _placeholder(value: str) -> bool:
    stripped = value.strip()
    return stripped.startswith("<") and stripped.endswith(">")


def main() -> None:
    output, counts = predict(_arguments())
    print(json.dumps({"output": str(output), **counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
