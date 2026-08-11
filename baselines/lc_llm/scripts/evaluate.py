#!/usr/bin/env python3
"""Evaluate LC-LLM predictions with strict coverage and top-1 ADE/FDE."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


BASELINE_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from lc_llm.config import (  # noqa: E402
    label_config_from_config,
    load_config,
    load_configured_split,
)
from lc_llm.data_adapter import future_to_local  # noqa: E402
from lc_llm.metrics import (  # noqa: E402
    CoverageError,
    ReferenceRecord,
    evaluate_predictions,
    load_prediction_jsonl,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to an LC-LLM JSON config")
    parser.add_argument(
        "--predictions", default=None, help="Override paths.predictions JSONL"
    )
    parser.add_argument("--output", default=None, help="Override paths.metrics JSON")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Diagnostic only: score the valid ID intersection instead of requiring full coverage",
    )
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> Path:
    """Join predictions to test labels by ID and write a JSON metric report."""

    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    data = _section(config, "data")
    evaluation = _section(config, "evaluation")
    paths = _section(config, "paths")
    expected_steps = int(data.get("future_steps", 50))
    sample_rate_hz = float(data.get("sample_rate_hz", 10.0))
    horizons = [float(value) for value in evaluation.get("horizons_seconds", [1, 2, 3, 4, 5])]
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

    # Prediction generation is deliberately not invoked here.  Future labels
    # are loaded only for this post hoc metric calculation.
    predictions = load_prediction_jsonl(
        prediction_path, expected_steps=expected_steps
    )
    samples = load_configured_split(config, "test", include_future=True)
    label_config = label_config_from_config(config)
    references: list[ReferenceRecord] = []
    for sample in samples:
        local_future = future_to_local(sample, expected_steps=expected_steps)
        if label_config.intention_label_mode == "phase_left_event":
            # The private CoT-TP preprocessing retains left-LC events only.
            # lane_status is a supervised phase label and is never written to
            # an inference prompt by this evaluator.
            intention = (
                "left_lane_change"
                if sample.observation.lane_status in {0, 1}
                else "keep_lane"
            )
        elif label_config.intention_label_mode == "future_lateral_displacement":
            lateral = float(local_future[-1, 1])
            threshold = float(label_config.lateral_threshold_m)
            if lateral > threshold:
                intention = "left_lane_change"
            elif lateral < -threshold:
                intention = "right_lane_change"
            else:
                intention = "keep_lane"
        else:  # guarded by config validation
            raise ValueError(
                f"unsupported intention mode: {label_config.intention_label_mode}"
            )
        references.append(
            ReferenceRecord(
                sample_id=sample.sample_id,
                intention=intention,
                trajectory=local_future,
            )
        )

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
            "baseline": "LC-LLM (adapted)",
            "status": "coverage_error",
            "prediction_mode": "top1",
            "coverage": exc.report,
            "message": str(exc),
        }
        _write_json(output_path, failure)
        raise

    result = {
        "baseline": "LC-LLM (adapted)",
        "status": "ok" if report["coverage"]["exact"] else "partial_diagnostic",
        "prediction_file": prediction_path.name,
        **report,
    }
    _write_json(output_path, result)
    _print_summary(result)
    return output_path


def _print_summary(report: Mapping[str, Any]) -> None:
    coverage = report["coverage"]
    print(
        f"coverage: {coverage['valid']}/{coverage['expected']} valid "
        f"({coverage['valid_fraction']:.2%})"
    )
    intention = report["intention"]
    print(
        f"intention: accuracy={intention['accuracy']:.6f} "
        f"macro-F1={intention['macro_f1']:.6f}"
    )
    for horizon, values in report["trajectory"].items():
        print(f"{horizon}: ADE={values['ADE']:.6f} FDE={values['FDE']:.6f}")


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
    stripped = value.strip()
    return stripped.startswith("<") and stripped.endswith(">")


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
