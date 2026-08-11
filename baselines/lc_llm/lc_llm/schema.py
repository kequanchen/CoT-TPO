"""Strict parser and formatter for adapted LC-LLM joint outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


INTENTION_LABELS = {
    0: "Keep lane",
    1: "Left lane change",
    2: "Right lane change",
}


class LCOutputParseError(ValueError):
    """Raised when generated text does not satisfy the released output schema."""


@dataclass(frozen=True)
class ParsedLCOutput:
    notable_features: tuple[str, ...]
    potential_behaviors: tuple[str, ...]
    intention: int
    trajectory: np.ndarray

    def as_json_dict(self) -> dict[str, Any]:
        return {
            "notable_features": list(self.notable_features),
            "potential_behaviors": list(self.potential_behaviors),
            "intention": self.intention,
            "trajectory": self.trajectory.tolist(),
        }


_BLOCK_RE = re.compile(
    r"\A\s*Thought:\s*(?P<thought>.*?)\s*Final\s+answer:\s*(?P<final>.*?)\s*\Z",
    re.IGNORECASE | re.DOTALL,
)
_NOTABLE_RE = re.compile(
    r"^\s*-\s*Notable\s+features?\s*:\s*(?P<value>\S.*?)\s*$",
    re.IGNORECASE,
)
_BEHAVIOR_RE = re.compile(
    r"^\s*-\s*Potential\s+behaviors?\s*:\s*(?P<value>\S.*?)\s*$",
    re.IGNORECASE,
)
_INTENTION_RE = re.compile(
    r"^\s*-\s*Intention\s*:\s*[\"']?(?P<index>[012])"
    r"(?:\s*:\s*(?P<label>[^\"']+?))?[\"']?\s*$",
    re.IGNORECASE,
)
_TRAJECTORY_RE = re.compile(
    r"^\s*-\s*Trajectory\s*:\s*(?P<value>.+)\s*\Z",
    re.IGNORECASE | re.DOTALL,
)


def _nonempty_lines(block: str) -> list[str]:
    return [line for line in block.splitlines() if line.strip()]


def _reject_constant(value: str) -> None:
    raise LCOutputParseError(f"trajectory contains non-finite JSON constant {value}")


def validate_trajectory(value: Any, *, expected_points: int = 50) -> np.ndarray:
    """Return a finite float32 ``[expected_points,2]`` trajectory."""

    if expected_points <= 0:
        raise ValueError("expected_points must be positive")
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise LCOutputParseError("trajectory must be a numeric array") from exc
    if array.shape != (expected_points, 2):
        raise LCOutputParseError(
            f"trajectory must contain exactly {expected_points} [x,y] points; got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise LCOutputParseError("trajectory coordinates must be finite")
    return np.array(array, dtype=np.float32, copy=True)


def parse_lc_llm_output(text: str, *, expected_points: int = 50) -> ParsedLCOutput:
    """Parse the paper-style Thought/Final-answer response without guessing."""

    if not isinstance(text, str) or not text.strip():
        raise LCOutputParseError("LC-LLM output must be a non-empty string")
    match = _BLOCK_RE.fullmatch(text)
    if match is None:
        raise LCOutputParseError("expected exactly one Thought and one Final answer block")

    notable: list[str] = []
    behaviors: list[str] = []
    for line in _nonempty_lines(match.group("thought")):
        feature_match = _NOTABLE_RE.fullmatch(line)
        behavior_match = _BEHAVIOR_RE.fullmatch(line)
        if feature_match:
            notable.append(feature_match.group("value").strip())
        elif behavior_match:
            behaviors.append(behavior_match.group("value").strip())
        else:
            raise LCOutputParseError(f"unrecognized Thought line: {line!r}")
    if not notable:
        raise LCOutputParseError("Thought must contain at least one Notable feature")
    if not behaviors:
        raise LCOutputParseError("Thought must contain at least one Potential behavior")

    final_lines = _nonempty_lines(match.group("final"))
    if len(final_lines) < 2:
        raise LCOutputParseError("Final answer must contain Intention and Trajectory")
    intention_match = _INTENTION_RE.fullmatch(final_lines[0])
    if intention_match is None:
        raise LCOutputParseError("invalid Intention line")
    intention = int(intention_match.group("index"))
    supplied_label = intention_match.group("label")
    if supplied_label is not None:
        normalized = " ".join(supplied_label.strip().split()).casefold()
        if normalized != INTENTION_LABELS[intention].casefold():
            raise LCOutputParseError(
                f"intention label does not match class {intention}: {supplied_label!r}"
            )

    trajectory_text = "\n".join(final_lines[1:])
    trajectory_match = _TRAJECTORY_RE.fullmatch(trajectory_text)
    if trajectory_match is None:
        raise LCOutputParseError("invalid Trajectory line")
    payload = trajectory_match.group("value").strip()
    if len(payload) >= 2 and payload[0] == payload[-1] and payload[0] in {'"', "'"}:
        payload = payload[1:-1].strip()
    try:
        coordinates = json.loads(payload, parse_constant=_reject_constant)
    except LCOutputParseError:
        raise
    except json.JSONDecodeError as exc:
        raise LCOutputParseError(f"Trajectory must be a JSON array: {exc.msg}") from exc
    trajectory = validate_trajectory(coordinates, expected_points=expected_points)
    return ParsedLCOutput(
        notable_features=tuple(notable),
        potential_behaviors=tuple(behaviors),
        intention=intention,
        trajectory=trajectory,
    )


def format_lc_llm_output(
    notable_features: Sequence[str],
    potential_behaviors: Sequence[str],
    intention: int,
    trajectory: Any,
    *,
    precision: int = 2,
) -> str:
    """Create the canonical supervised answer consumed by the strict parser."""

    if intention not in INTENTION_LABELS:
        raise ValueError("intention must be 0, 1, or 2")
    if precision < 0 or precision > 8:
        raise ValueError("precision must be between 0 and 8")
    features = tuple(str(item).strip() for item in notable_features)
    behaviors = tuple(str(item).strip() for item in potential_behaviors)
    if not features or any(not item for item in features):
        raise ValueError("at least one non-empty notable feature is required")
    if not behaviors or any(not item for item in behaviors):
        raise ValueError("at least one non-empty potential behavior is required")
    array = validate_trajectory(trajectory, expected_points=50)
    rounded = [[round(float(x), precision), round(float(y), precision)] for x, y in array]
    serialized = json.dumps(rounded, ensure_ascii=True, separators=(",", ":"))
    lines = ["Thought:"]
    lines.extend(f"- Notable feature: {item}" for item in features)
    lines.extend(f"- Potential behavior: {item}" for item in behaviors)
    lines.extend(
        [
            "Final answer:",
            f'- Intention: "{intention}: {INTENTION_LABELS[intention]}"',
            f'- Trajectory: "{serialized}"',
        ]
    )
    return "\n".join(lines)


def parse_generated_answer(text: str, *, expected_points: int = 50) -> ParsedLCOutput:
    """Compatibility alias used by evaluation code."""

    return parse_lc_llm_output(text, expected_points=expected_points)


def parse_prediction_record(
    record: Mapping[str, Any], *, expected_points: int = 50
) -> ParsedLCOutput:
    """Validate either a raw-output record or parsed prediction fields."""

    if "raw_output" in record:
        return parse_lc_llm_output(str(record["raw_output"]), expected_points=expected_points)
    if "intention" not in record or "trajectory" not in record:
        raise LCOutputParseError("prediction record needs raw_output or intention+trajectory")
    intention = int(record["intention"])
    if intention not in INTENTION_LABELS:
        raise LCOutputParseError("prediction intention must be 0, 1, or 2")
    trajectory = validate_trajectory(record["trajectory"], expected_points=expected_points)
    return ParsedLCOutput((), (), intention, trajectory)
