"""Strict evaluation utilities for the paper-reconstructed LC-LLM baseline.

Predictions are joined to references by ``sample_id``.  A failed language-model
parse is an invalid prediction, not a sample that may be silently discarded.
The default evaluation therefore requires one valid prediction for every test
sample before reporting intention or trajectory metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


INTENTION_LABELS = ("keep_lane", "left_lane_change", "right_lane_change")


class CoverageError(ValueError):
    """Raised when strict evaluation does not have exact, valid coverage."""

    def __init__(self, message: str, report: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.report = dict(report)


@dataclass(frozen=True)
class ReferenceRecord:
    """Ground-truth intention and target-centered future trajectory."""

    sample_id: str
    intention: str
    trajectory: np.ndarray


@dataclass(frozen=True)
class PredictionRecord:
    """One successfully parsed top-level JSONL prediction."""

    sample_id: str
    intention: str
    trajectory: np.ndarray


@dataclass(frozen=True)
class PredictionFile:
    """All attempted predictions, including explicit generation failures."""

    attempted_ids: frozenset[str]
    valid: Mapping[str, PredictionRecord]
    invalid: Mapping[str, str]


def normalize_intention(value: Any) -> str:
    """Normalize common LC-LLM intention spellings to three canonical labels."""

    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        numeric = int(value)
        if numeric in (0, 1, 2):
            return INTENTION_LABELS[numeric]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("intention must be a non-empty string or one of 0, 1, 2")
    compact = " ".join(value.strip().lower().replace("-", " ").replace("_", " ").split())
    aliases = {
        "0": "keep_lane",
        "keep": "keep_lane",
        "keep lane": "keep_lane",
        "lane keeping": "keep_lane",
        "lk": "keep_lane",
        "1": "left_lane_change",
        "left": "left_lane_change",
        "left lane change": "left_lane_change",
        "change left": "left_lane_change",
        "llc": "left_lane_change",
        "2": "right_lane_change",
        "right": "right_lane_change",
        "right lane change": "right_lane_change",
        "change right": "right_lane_change",
        "rlc": "right_lane_change",
    }
    if compact not in aliases:
        raise ValueError(f"unsupported intention label: {value!r}")
    return aliases[compact]


def as_trajectory(value: Any, *, expected_steps: int) -> np.ndarray:
    """Return a finite ``[expected_steps, 2]`` trajectory."""

    if expected_steps <= 0:
        raise ValueError("expected_steps must be positive")
    try:
        trajectory = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("trajectory must be a numeric array") from exc
    if trajectory.shape != (expected_steps, 2):
        raise ValueError(
            f"trajectory must have shape ({expected_steps}, 2), got {trajectory.shape}"
        )
    if not np.isfinite(trajectory).all():
        raise ValueError("trajectory contains non-finite coordinates")
    return trajectory


def load_prediction_jsonl(
    path: str | Path,
    *,
    expected_steps: int,
) -> PredictionFile:
    """Load top-level predictions while retaining failed attempts for coverage.

    Each non-empty line must be a JSON object with a unique ``sample_id`` and a
    ``status``.  Successful records require top-level ``intention`` and
    ``trajectory`` fields.  ``raw_output`` may be retained for auditing but is
    never reparsed here, which keeps metric calculation deterministic.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"prediction JSONL not found: {source}")
    attempted: set[str] = set()
    valid: dict[str, PredictionRecord] = {}
    invalid: dict[str, str] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {source}:{line_number}: {exc}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"prediction at {source}:{line_number} must be a JSON object")
            sample_id = _sample_id(payload.get("sample_id"))
            if sample_id in attempted:
                raise ValueError(
                    f"duplicate sample_id at {source}:{line_number}: {sample_id}"
                )
            attempted.add(sample_id)
            status = payload.get("status", "ok")
            if status != "ok":
                error = payload.get("error")
                invalid[sample_id] = str(error or f"generation status is {status!r}")
                continue
            try:
                valid[sample_id] = PredictionRecord(
                    sample_id=sample_id,
                    intention=normalize_intention(payload.get("intention")),
                    trajectory=as_trajectory(
                        payload.get("trajectory"), expected_steps=expected_steps
                    ),
                )
            except ValueError as exc:
                invalid[sample_id] = str(exc)
    if not attempted:
        raise ValueError(f"no predictions found in {source}")
    return PredictionFile(frozenset(attempted), valid, invalid)


def evaluate_predictions(
    references: Iterable[ReferenceRecord | Mapping[str, Any]],
    predictions: PredictionFile | Iterable[PredictionRecord | Mapping[str, Any]],
    *,
    expected_steps: int,
    sample_rate_hz: float,
    horizons_seconds: Sequence[float],
    require_full_coverage: bool = True,
) -> dict[str, Any]:
    """Compute exact-match intention metrics and top-1 ADE/FDE.

    ``trajectory`` is one generated sequence per sample; no oracle selection is
    performed.  In strict mode the set of valid prediction IDs must equal the
    reference IDs exactly.  In diagnostic non-strict mode, metrics use only the
    valid intersection and the incomplete coverage remains explicit.
    """

    reference_map = _reference_map(references, expected_steps=expected_steps)
    prediction_file = _prediction_file(predictions, expected_steps=expected_steps)
    expected_ids = set(reference_map)
    attempted_expected = expected_ids & set(prediction_file.attempted_ids)
    valid_ids = expected_ids & set(prediction_file.valid)
    missing_ids = expected_ids - set(prediction_file.attempted_ids)
    invalid_ids = expected_ids & set(prediction_file.invalid)
    extra_ids = set(prediction_file.attempted_ids) - expected_ids
    expected_count = len(expected_ids)
    coverage = {
        "expected": expected_count,
        "attempted": len(attempted_expected),
        "valid": len(valid_ids),
        "attempted_fraction": len(attempted_expected) / expected_count,
        "valid_fraction": len(valid_ids) / expected_count,
        "missing_sample_ids": sorted(missing_ids),
        "invalid_sample_ids": sorted(invalid_ids),
        "extra_sample_ids": sorted(extra_ids),
        "invalid_reasons": {
            sample_id: prediction_file.invalid[sample_id]
            for sample_id in sorted(invalid_ids)
        },
        "require_full_coverage": bool(require_full_coverage),
    }
    exact = (
        len(valid_ids) == expected_count
        and not missing_ids
        and not invalid_ids
        and not extra_ids
    )
    coverage["exact"] = exact
    if require_full_coverage and not exact:
        raise CoverageError(
            "strict coverage failed: every test sample must have exactly one valid "
            "prediction and no extra prediction IDs are allowed",
            coverage,
        )
    if not valid_ids:
        raise CoverageError("no valid matched predictions are available", coverage)

    ordered_ids = sorted(valid_ids)
    truth_intentions = [reference_map[item].intention for item in ordered_ids]
    predicted_intentions = [prediction_file.valid[item].intention for item in ordered_ids]
    intention_metrics = intention_classification_metrics(
        truth_intentions, predicted_intentions
    )
    true_trajectories = np.stack(
        [reference_map[item].trajectory for item in ordered_ids], axis=0
    )
    predicted_trajectories = np.stack(
        [prediction_file.valid[item].trajectory for item in ordered_ids], axis=0
    )
    trajectory_metrics = top1_ade_fde(
        predicted_trajectories,
        true_trajectories,
        sample_rate_hz=sample_rate_hz,
        horizons_seconds=horizons_seconds,
    )
    return {
        "prediction_mode": "top1",
        "coverage": coverage,
        "intention": intention_metrics,
        "trajectory": trajectory_metrics,
    }


def intention_classification_metrics(
    truth: Sequence[Any],
    prediction: Sequence[Any],
    *,
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute accuracy, per-class metrics, and macro F1 without sklearn."""

    if len(truth) != len(prediction) or not truth:
        raise ValueError("truth and prediction must be non-empty and equally sized")
    y_true = [normalize_intention(value) for value in truth]
    y_pred = [normalize_intention(value) for value in prediction]
    if labels is None:
        observed = set(y_true) | set(y_pred)
        canonical_labels = tuple(label for label in INTENTION_LABELS if label in observed)
    else:
        canonical_labels = tuple(normalize_intention(label) for label in labels)
        if len(set(canonical_labels)) != len(canonical_labels):
            raise ValueError("labels must be unique")
    per_class: dict[str, dict[str, float | int]] = {}
    for label in canonical_labels:
        tp = sum(a == label and b == label for a, b in zip(y_true, y_pred))
        fp = sum(a != label and b == label for a, b in zip(y_true, y_pred))
        fn = sum(a == label and b != label for a, b in zip(y_true, y_pred))
        support = sum(a == label for a in y_true)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return {
        "accuracy": sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true),
        "macro_f1": sum(float(per_class[label]["f1"]) for label in canonical_labels)
        / len(canonical_labels),
        "samples": len(y_true),
        "labels": list(canonical_labels),
        "per_class": per_class,
    }


def top1_ade_fde(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    sample_rate_hz: float,
    horizons_seconds: Sequence[float],
) -> dict[str, dict[str, float | int]]:
    """Compute Euclidean ADE and FDE for one trajectory per sample."""

    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if prediction.ndim != 3 or prediction.shape[-1] != 2:
        raise ValueError("prediction must have shape [samples, future_steps, 2]")
    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth trajectory arrays must have equal shape")
    if prediction.shape[0] == 0:
        raise ValueError("trajectory arrays cannot be empty")
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("trajectory arrays contain non-finite coordinates")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive and finite")
    horizons = [float(value) for value in horizons_seconds]
    if not horizons or len(set(horizons)) != len(horizons):
        raise ValueError("horizons_seconds must be non-empty and unique")
    distances = np.linalg.norm(prediction - truth, axis=-1)
    result: dict[str, dict[str, float | int]] = {}
    for seconds in horizons:
        steps = int(round(seconds * sample_rate_hz))
        if seconds <= 0 or steps <= 0 or steps > prediction.shape[1]:
            raise ValueError(
                f"horizon {seconds:g}s maps to invalid trajectory step {steps}"
            )
        result[f"{seconds:g}s"] = {
            "ADE": float(distances[:, :steps].mean()),
            "FDE": float(distances[:, steps - 1].mean()),
            "samples": int(prediction.shape[0]),
            "steps": steps,
        }
    return result


def _sample_id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("sample_id must be a non-empty string")
    return value.strip()


def _reference_map(
    records: Iterable[ReferenceRecord | Mapping[str, Any]], *, expected_steps: int
) -> dict[str, ReferenceRecord]:
    result: dict[str, ReferenceRecord] = {}
    for item in records:
        if isinstance(item, ReferenceRecord):
            sample_id = _sample_id(item.sample_id)
            record = ReferenceRecord(
                sample_id,
                normalize_intention(item.intention),
                as_trajectory(item.trajectory, expected_steps=expected_steps),
            )
        elif isinstance(item, Mapping):
            sample_id = _sample_id(item.get("sample_id"))
            record = ReferenceRecord(
                sample_id,
                normalize_intention(item.get("intention")),
                as_trajectory(item.get("trajectory"), expected_steps=expected_steps),
            )
        else:
            raise TypeError("reference records must be mappings or ReferenceRecord objects")
        if sample_id in result:
            raise ValueError(f"duplicate reference sample_id: {sample_id}")
        result[sample_id] = record
    if not result:
        raise ValueError("references cannot be empty")
    return result


def _prediction_file(
    records: PredictionFile | Iterable[PredictionRecord | Mapping[str, Any]],
    *,
    expected_steps: int,
) -> PredictionFile:
    if isinstance(records, PredictionFile):
        return records
    attempted: set[str] = set()
    valid: dict[str, PredictionRecord] = {}
    for item in records:
        if isinstance(item, PredictionRecord):
            sample_id = _sample_id(item.sample_id)
            record = PredictionRecord(
                sample_id,
                normalize_intention(item.intention),
                as_trajectory(item.trajectory, expected_steps=expected_steps),
            )
        elif isinstance(item, Mapping):
            sample_id = _sample_id(item.get("sample_id"))
            record = PredictionRecord(
                sample_id,
                normalize_intention(item.get("intention")),
                as_trajectory(item.get("trajectory"), expected_steps=expected_steps),
            )
        else:
            raise TypeError("predictions must be mappings or PredictionRecord objects")
        if sample_id in attempted:
            raise ValueError(f"duplicate prediction sample_id: {sample_id}")
        attempted.add(sample_id)
        valid[sample_id] = record
    if not attempted:
        raise ValueError("predictions cannot be empty")
    return PredictionFile(frozenset(attempted), valid, {})
