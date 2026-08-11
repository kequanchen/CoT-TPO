#!/usr/bin/env python3
"""Evaluate adapted Direct LLM outputs with strict top-1 ADE/FDE."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


BASELINE_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from direct_llm.config import load_config, load_configured_split  # noqa: E402
from direct_llm.data_adapter import future_to_local  # noqa: E402
from direct_llm.metrics import (  # noqa: E402
    CoverageError,
    ReferenceRecord,
    evaluate_predictions,
    load_prediction_jsonl,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a Direct LLM config")
    parser.add_argument(
        "--predictions", default=None, help="Override paths.predictions JSONL"
    )
    parser.add_argument("--output", default=None, help="Override paths.metrics JSON")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Diagnostic only: score the valid ID intersection instead of full coverage",
    )
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> Path:
    config = load_config(Path(args.config).expanduser().resolve())
    data = _section(config, "data")
    paths = _section(config, "paths")
    evaluation = _section(config, "evaluation")
    expected_steps = int(data.get("future_steps", 50))
    sample_rate_hz = float(data.get("sample_rate_hz", 10.0))
    horizons = [
        float(value)
        for value in evaluation.get("horizons_seconds", [1, 2, 3, 4, 5])
    ]
    if str(evaluation.get("prediction_mode", "top1")).lower() != "top1":
        raise ValueError("evaluation.prediction_mode must be 'top1'")
    require_full = bool(evaluation.get("require_full_coverage", True))
    if args.allow_partial:
        require_full = False

    prediction_path = _configured_path(
        args.predictions or paths.get("predictions"), "paths.predictions"
    )
    output_path = _configured_path(
        args.output or paths.get("metrics"), "paths.metrics", create_parent=True
    )

    # This script never invokes the LLM.  It first loads the completed output
    # file and only then accesses labelled test futures for metric calculation.
    predictions = load_prediction_jsonl(
        prediction_path, expected_steps=expected_steps
    )
    samples = load_configured_split(config, "test", include_future=True)
    references = [
        ReferenceRecord(
            sample_id=sample.sample_id,
            trajectory=future_to_local(sample, expected_steps=expected_steps),
        )
        for sample in samples
    ]

    try:
        report = evaluate_predictions(
            references,
            predictions,
            expected_steps=expected_steps,
            sample_rate_hz=sample_rate_hz,
            horizons_seconds=horizons,
            require_full_coverage=require_full,
        )
    except CoverageError as exc:
        failure = {
            "baseline": "Direct LLM (adapted)",
            "status": "coverage_error",
            "prediction_mode": "top1",
            "coverage": exc.report,
            "message": str(exc),
        }
        _write_json(output_path, failure)
        raise

    result = {
        "baseline": "Direct LLM (adapted)",
        "status": "ok" if report["coverage"]["exact"] else "partial_diagnostic",
        "prediction_file": prediction_path.name,
        **report,
    }
    _write_json(output_path, result)
    coverage = result["coverage"]
    print(
        f"coverage: {coverage['valid']}/{coverage['expected']} valid "
        f"({coverage['valid_fraction']:.2%})"
    )
    for horizon, values in result["trajectory"].items():
        print(f"{horizon}: ADE={values['ADE']:.6f} FDE={values['FDE']:.6f}")
    return output_path


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = config.get(name)
    if not isinstance(section, Mapping):
        raise ValueError(f"missing configuration object: {name}")
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
    text = value.strip()
    return text.startswith("<") and text.endswith(">")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def main() -> None:
    path = evaluate(_arguments())
    print(path)


if __name__ == "__main__":
    main()
