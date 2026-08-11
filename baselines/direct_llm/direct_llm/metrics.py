"""Strict top-1 trajectory metrics for the adapted Direct LLM baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


class CoverageError(ValueError):
    """Raised when predictions do not exactly and validly cover the test set."""

    def __init__(self, message: str, report: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.report = dict(report)


@dataclass(frozen=True)
class ReferenceRecord:
    sample_id: str
    trajectory: np.ndarray


@dataclass(frozen=True)
class PredictionRecord:
    sample_id: str
    trajectory: np.ndarray


@dataclass(frozen=True)
class PredictionFile:
    """Attempted IDs plus valid and failed prediction records."""

    attempted_ids: frozenset[str]
    valid: Mapping[str, PredictionRecord]
    invalid: Mapping[str, str]


def as_trajectory(value: Any, *, expected_steps: int) -> np.ndarray:
    """Return a finite trajectory with exactly ``expected_steps`` xy points."""

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
    """Load top-level JSONL predictions and retain failed attempts.

    Successful lines require ``sample_id``, ``status: ok``, and a top-level
    ``trajectory``.  API and parse failures remain part of coverage instead of
    disappearing from the evaluation denominator.
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
                raise ValueError(f"prediction at {source}:{line_number} must be an object")
            sample_id = _sample_id(payload.get("sample_id"))
            if sample_id in attempted:
                raise ValueError(
                    f"duplicate sample_id at {source}:{line_number}: {sample_id}"
                )
            attempted.add(sample_id)
            status = payload.get("status", "ok")
            if status != "ok":
                invalid[sample_id] = str(
                    payload.get("error") or f"prediction status is {status!r}"
                )
                continue
            try:
                trajectory = as_trajectory(
                    payload.get("trajectory"), expected_steps=expected_steps
                )
            except ValueError as exc:
                invalid[sample_id] = str(exc)
                continue
            valid[sample_id] = PredictionRecord(sample_id, trajectory)
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
    """Join by ID and calculate one-trajectory ADE/FDE at each horizon."""

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
    truth = np.stack([reference_map[item].trajectory for item in ordered_ids])
    prediction = np.stack(
        [prediction_file.valid[item].trajectory for item in ordered_ids]
    )
    return {
        "prediction_mode": "top1",
        "coverage": coverage,
        "trajectory": top1_ade_fde(
            prediction,
            truth,
            sample_rate_hz=sample_rate_hz,
            horizons_seconds=horizons_seconds,
        ),
    }


def top1_ade_fde(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    sample_rate_hz: float,
    horizons_seconds: Sequence[float],
) -> dict[str, dict[str, float | int]]:
    """Compute Euclidean ADE/FDE for exactly one generated path per sample."""

    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if prediction.ndim != 3 or prediction.shape[-1] != 2:
        raise ValueError("prediction must have shape [samples, future_steps, 2]")
    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth must have equal shapes")
    if prediction.shape[0] == 0:
        raise ValueError("trajectory arrays cannot be empty")
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("trajectory arrays contain non-finite values")
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
                as_trajectory(item.trajectory, expected_steps=expected_steps),
            )
        elif isinstance(item, Mapping):
            sample_id = _sample_id(item.get("sample_id"))
            record = ReferenceRecord(
                sample_id,
                as_trajectory(item.get("trajectory"), expected_steps=expected_steps),
            )
        else:
            raise TypeError("references must be mappings or ReferenceRecord objects")
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
                as_trajectory(item.trajectory, expected_steps=expected_steps),
            )
        elif isinstance(item, Mapping):
            sample_id = _sample_id(item.get("sample_id"))
            record = PredictionRecord(
                sample_id,
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
