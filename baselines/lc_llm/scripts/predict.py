#!/usr/bin/env python3
"""Run safe, batched, restartable inference with an LC-LLM LoRA adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping


BASELINE_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate joint reasoning, intention, and trajectory outputs with LC-LLM."
    )
    parser.add_argument("--config", required=True, help="Path to an LC-LLM JSON configuration")
    parser.add_argument(
        "--input-jsonl",
        default=None,
        help="Override paths.test_jsonl/paths.inference_jsonl with a prepared prompt JSONL",
    )
    parser.add_argument("--adapter-dir", default=None, help="Override paths.adapter_dir")
    parser.add_argument("--output", default=None, help="Override paths.predictions")
    parser.add_argument("--base-model", default=None, help="Require this adapter base-model identity")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip sample IDs already present in the output (default: true)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-resumed output")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at the first generation error")
    parser.add_argument("--seed", type=int, default=None, help="Override generation.seed")
    return parser.parse_args()


def predict(args: argparse.Namespace) -> tuple[Path, dict[str, int]]:
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")
    config = _load_json_config(Path(args.config).expanduser().resolve())
    paths = _section(config, "paths")
    model_cfg = _section(config, "model")
    generation_cfg = _section(config, "generation")
    if bool(model_cfg.get("trust_remote_code", False)):
        raise ValueError("model.trust_remote_code must remain false for this public baseline")

    from lc_llm.generation import (
        GenerationSettings,
        generate_text_batch,
        run_batched_prediction,
    )
    from lc_llm.modeling import (
        ModelLoadSettings,
        load_adapter_for_inference,
        load_tokenizer,
        set_global_seed,
    )
    from lc_llm.schema import parse_lc_llm_output
    from lc_llm.sft import load_jsonl_records

    input_value = args.input_jsonl or paths.get("test_jsonl") or paths.get("inference_jsonl")
    input_path = _configured_path(input_value, "paths.test_jsonl")
    adapter_path = _configured_path(
        args.adapter_dir or paths.get("adapter_dir"),
        "paths.adapter_dir",
    )
    output_path = _configured_path(
        args.output or paths.get("predictions"),
        "paths.predictions",
        create_parent=True,
    )
    expected_base = args.base_model
    if expected_base is None:
        configured_base = model_cfg.get("base_model")
        if isinstance(configured_base, str) and configured_base.strip() and not _placeholder(
            configured_base
        ):
            expected_base = configured_base.strip()

    seed = int(args.seed if args.seed is not None else generation_cfg.get("seed", 42))
    set_global_seed(seed, deterministic=bool(generation_cfg.get("deterministic", False)))
    settings = GenerationSettings(
        batch_size=int(generation_cfg.get("batch_size", 4)),
        max_input_tokens=int(
            generation_cfg.get(
                "max_input_tokens",
                int(model_cfg.get("max_seq_length", 4096))
                - int(generation_cfg.get("max_new_tokens", 1024)),
            )
        ),
        max_new_tokens=int(generation_cfg.get("max_new_tokens", 1024)),
        do_sample=bool(generation_cfg.get("do_sample", False)),
        temperature=float(generation_cfg.get("temperature", 0.7)),
        top_p=float(generation_cfg.get("top_p", 0.9)),
        num_beams=int(generation_cfg.get("num_beams", 1)),
    )
    max_sequence_length = int(model_cfg.get("max_seq_length", 4096))
    settings.validate_context_window(max_sequence_length)
    load_settings = ModelLoadSettings(
        load_in_4bit=bool(model_cfg.get("load_in_4bit", False)),
        load_in_8bit=bool(model_cfg.get("load_in_8bit", False)),
        torch_dtype=str(model_cfg.get("torch_dtype", "auto")),
        device_map=model_cfg.get("device_map", "auto"),
        local_files_only=bool(model_cfg.get("local_files_only", False)),
    )
    model, adapter_base = load_adapter_for_inference(
        adapter_path,
        expected_base_model=expected_base,
        settings=load_settings,
    )
    tokenizer_source = (
        str(adapter_path)
        if (adapter_path / "tokenizer_config.json").is_file()
        else adapter_base
    )
    tokenizer = load_tokenizer(
        tokenizer_source,
        local_files_only=bool(model_cfg.get("local_files_only", False)),
    )
    records = load_jsonl_records(
        input_path,
        require_answer=False,
        max_records=args.max_samples,
    )
    expected_points = int(_optional_section(config, "data").get("future_steps", 50))

    def model_predictor(batch):
        # Deriving the RNG seed from stable sample IDs makes sampled decoding
        # reproducible even when a completed prefix is skipped after restart.
        set_global_seed(
            _batch_seed(seed, [record.sample_id for record in batch]),
            deterministic=bool(generation_cfg.get("deterministic", False)),
        )
        return generate_text_batch(model, tokenizer, batch, settings)

    def strict_parser(text: str):
        return parse_lc_llm_output(text, expected_points=expected_points)

    counters = run_batched_prediction(
        records,
        output_path,
        model_predictor,
        batch_size=settings.batch_size,
        resume=args.resume,
        overwrite=args.overwrite,
        parser=strict_parser,
        fail_fast=args.fail_fast,
    )
    return output_path, counters


def _load_json_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"LC-LLM configuration not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("LC-LLM configuration must be a JSON object")
    return payload


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = config.get(name)
    if not isinstance(section, Mapping):
        raise ValueError(f"missing configuration object: {name}")
    return section


def _optional_section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = config.get(name, {})
    if not isinstance(section, Mapping):
        raise ValueError(f"configuration section {name} must be an object")
    return section


def _configured_path(value: Any, field: str, *, create_parent: bool = False) -> Path:
    if not isinstance(value, str) or not value.strip() or _placeholder(value):
        raise ValueError(f"{field} must be set in a private configuration or CLI argument")
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


def _batch_seed(base_seed: int, sample_ids: list[str]) -> int:
    digest = hashlib.sha256()
    digest.update(str(base_seed).encode("ascii"))
    for sample_id in sample_ids:
        encoded = sample_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:4], "big")


def main() -> None:
    output, counters = predict(_arguments())
    print(json.dumps({"output": str(output), **counters}, ensure_ascii=False))


if __name__ == "__main__":
    main()
