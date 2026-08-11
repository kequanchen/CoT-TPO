"""Strict, non-repairing output schema for the adapted Direct LLM baseline."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping

import numpy as np


class DirectOutputParseError(ValueError):
    """Raised when an LLM response is not the exact released JSON contract."""


@dataclass(frozen=True)
class ParsedDirectOutput:
    trajectory: np.ndarray

    @property
    def future_trajectory(self) -> np.ndarray:
        return self.trajectory

    def as_json_dict(self) -> dict[str, Any]:
        return {"future_trajectory": self.trajectory.tolist()}


def _reject_constant(value: str) -> None:
    raise DirectOutputParseError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DirectOutputParseError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def validate_trajectory(value: Any, *, expected_points: int = 50) -> np.ndarray:
    """Validate exactly one finite ``[expected_points,2]`` coordinate sequence."""

    if expected_points <= 0:
        raise ValueError("expected_points must be positive")
    if not isinstance(value, list) or len(value) != expected_points:
        count = len(value) if isinstance(value, list) else "non-list"
        raise DirectOutputParseError(
            f"future_trajectory must contain exactly {expected_points} points; got {count}"
        )
    normalized: list[list[float]] = []
    for index, point in enumerate(value):
        if not isinstance(point, list) or len(point) != 2:
            raise DirectOutputParseError(f"point {index} must be a two-item JSON array")
        pair: list[float] = []
        for coordinate in point:
            if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
                raise DirectOutputParseError(f"point {index} coordinates must be JSON numbers")
            number = float(coordinate)
            if not math.isfinite(number):
                raise DirectOutputParseError(f"point {index} coordinates must be finite")
            pair.append(number)
        normalized.append(pair)
    return np.asarray(normalized, dtype=np.float32)


def parse_direct_output(text: str, *, expected_points: int = 50) -> ParsedDirectOutput:
    """Parse one bare JSON object without extracting, repairing, or resampling."""

    if not isinstance(text, str) or not text.strip():
        raise DirectOutputParseError("Direct LLM output must be a non-empty string")
    try:
        payload = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except DirectOutputParseError:
        raise
    except json.JSONDecodeError as exc:
        raise DirectOutputParseError(
            f"output must be one bare JSON object: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise DirectOutputParseError("output must be a JSON object")
    if set(payload) != {"future_trajectory"}:
        raise DirectOutputParseError(
            'output must contain only the key "future_trajectory"'
        )
    return ParsedDirectOutput(
        validate_trajectory(payload["future_trajectory"], expected_points=expected_points)
    )


def parse_direct_llm_output(
    text: str, *, expected_points: int = 50
) -> ParsedDirectOutput:
    """Explicitly named compatibility alias."""

    return parse_direct_output(text, expected_points=expected_points)


def parse_generated_answer(
    text: str, *, expected_points: int = 50
) -> ParsedDirectOutput:
    """Compatibility alias used by the generic generation pipeline."""

    return parse_direct_output(text, expected_points=expected_points)


def format_direct_output(
    trajectory: Any,
    *,
    precision: int = 2,
    expected_points: int = 50,
) -> str:
    """Serialize a known trajectory into the canonical single-object response."""

    if not 0 <= precision <= 8:
        raise ValueError("precision must be between 0 and 8")
    # Formatter accepts NumPy inputs, while parser deliberately accepts JSON
    # lists only.  Convert first, then pass through the same strict validator.
    try:
        value = np.asarray(trajectory).tolist()
    except Exception as exc:  # pragma: no cover - defensive non-array object
        raise DirectOutputParseError("trajectory cannot be converted to coordinates") from exc
    array = validate_trajectory(value, expected_points=expected_points)
    rounded = [
        [round(float(forward), precision), round(float(left), precision)]
        for forward, left in array
    ]
    return json.dumps(
        {"future_trajectory": rounded},
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )


def parse_prediction_record(
    record: Mapping[str, Any], *, expected_points: int = 50
) -> ParsedDirectOutput:
    """Validate either a raw response or an already expanded prediction row."""

    if "raw_output" in record:
        return parse_direct_output(str(record["raw_output"]), expected_points=expected_points)
    if "trajectory" in record:
        trajectory = record["trajectory"]
    elif "future_trajectory" in record:
        trajectory = record["future_trajectory"]
    else:
        raise DirectOutputParseError(
            "prediction record needs raw_output, trajectory, or future_trajectory"
        )
    value = np.asarray(trajectory).tolist()
    return ParsedDirectOutput(validate_trajectory(value, expected_points=expected_points))
