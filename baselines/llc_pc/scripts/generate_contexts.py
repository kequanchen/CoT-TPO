#!/usr/bin/env python3
"""Generate observation-only LLC-PC prompts, maps, and structured contexts."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


BASELINE_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from llc_pc import (  # noqa: E402
    build_prompt,
    encode_context,
    load_config,
    load_configured_split,
    parse_llm_response,
    render_tc_map_png,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render observation-only TC-maps and optionally query an OpenAI-compatible VLM."
    )
    parser.add_argument("--config", required=True, help="Path to a local LLC-PC JSON config")
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="train"
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Write prompts and maps without making external API calls",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _artifact_stem(sample_key: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_key).strip("_.-")[:48]
    digest = hashlib.sha256(sample_key.encode("utf-8")).hexdigest()[:12]
    return f"{readable or 'sample'}-{digest}"


def _vlm_client(config: dict[str, Any]):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("install the 'openai' package before enabling VLM calls") from exc
    llm = config["llm"]
    key_name = str(llm["api_key_env"])
    api_key = os.environ.get(key_name)
    if not api_key:
        raise RuntimeError(f"required API credential environment variable is unset: {key_name}")
    base_name = str(llm.get("base_url_env", "")).strip()
    base_url = os.environ.get(base_name) if base_name else None
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _query_vlm(client: Any, config: dict[str, Any], prompt: str, png: bytes) -> str:
    llm = config["llm"]
    model = str(llm["model"])
    if model.startswith("YOUR_") or (model.startswith("<") and model.endswith(">")):
        raise ValueError("llm.model is still a public placeholder")
    image_url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        temperature=float(llm.get("temperature", 0.7)),
        max_tokens=int(llm.get("max_tokens", 1200)),
        seed=int(llm.get("seed", 42)),
    )
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("the VLM returned an empty response")
    return content


def generate(args: argparse.Namespace) -> Path:
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    config = load_config(args.config)
    # This is the central leakage guard: y_future is not even loaded here.
    samples = load_configured_split(config, args.split, include_future=False)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    output_dir = Path(config["paths"]["raw_context_dir"]) / args.split
    prompt_dir = output_dir / "prompts"
    image_dir = output_dir / "tc_maps"
    manifest_path = output_dir / "contexts.jsonl"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"output already exists: {manifest_path}; pass --overwrite to replace it"
        )
    prompt_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    client = None if args.skip_llm else _vlm_client(config)

    failures = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for sample in samples:
            observation = sample.observation
            stem = _artifact_stem(observation.sample_key)
            prompt_path = prompt_dir / f"{stem}.txt"
            image_path = image_dir / f"{stem}.png"
            prompt = build_prompt(observation)
            png = render_tc_map_png(observation, image_path)
            prompt_path.write_text(prompt, encoding="utf-8")
            record: dict[str, Any] = {
                "source_split": args.split,
                "sample_id": observation.sample_key,
                "event_id": str(observation.scenario_id),
                "prompt_file": str(prompt_path.relative_to(output_dir)),
                "tc_map_file": str(image_path.relative_to(output_dir)),
                "status": "dry_run" if args.skip_llm else "pending",
            }
            if not args.skip_llm:
                try:
                    assert client is not None
                    raw_response = _query_vlm(client, config, prompt, png)
                    parsed = parse_llm_response(raw_response)
                    record.update(
                        status="ok",
                        response=parsed.as_json_dict(),
                        context_vector=encode_context(parsed).tolist(),
                    )
                except Exception as exc:  # preserve batch progress and make failures auditable
                    failures += 1
                    record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")

    if failures:
        raise RuntimeError(
            f"{failures} VLM response(s) failed validation; inspect {manifest_path}"
        )
    return manifest_path


def main() -> None:
    path = generate(_arguments())
    print(path)


if __name__ == "__main__":
    main()
