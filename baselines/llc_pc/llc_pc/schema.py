"""Strict LLC-PC JSON schema and deterministic 17-dimensional encoding."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Union

import numpy as np


INTENTIONS = (
    "STATIONARY",
    "STRAIGHT",
    "STRAIGHT_LEFT",
    "STRAIGHT_RIGHT",
    "LEFT_TURN",
    "RIGHT_TURN",
    "LEFT_U_TURN",
    "RIGHT_U_TURN",
)
AFFORDANCES = (
    "SLOW_ALLOW",
    "ACCELERATE_ALLOW",
    "LEFT_ALLOW",
    "RIGHT_ALLOW",
)
SCENARIOS = (
    "INTERSECTION",
    "ON_STRAIGHT_ROAD",
    "PARKING_LOT",
    "ON_ROADSIDE",
    "UNSURE",
)

CONTEXT_DIM = len(INTENTIONS) + len(AFFORDANCES) + len(SCENARIOS)
INTENTION_SLICE = slice(0, len(INTENTIONS))
AFFORDANCE_SLICE = slice(len(INTENTIONS), len(INTENTIONS) + len(AFFORDANCES))
SCENARIO_SLICE = slice(len(INTENTIONS) + len(AFFORDANCES), CONTEXT_DIM)

_FIELDS = {
    "Situation Understanding",
    "Reasoning",
    "Actions",
    "Affordance",
    "Scenario_name",
}


class ContextParseError(ValueError):
    """Raised when an LLM response violates the released context schema."""


@dataclass(frozen=True)
class ParsedContext:
    situation_understanding: str
    reasoning: str
    actions: tuple[str, ...]
    affordances: tuple[str, ...]
    scenario: str

    def as_json_dict(self) -> dict[str, Any]:
        return {
            "Situation Understanding": self.situation_understanding,
            "Reasoning": self.reasoning,
            "Actions": list(self.actions),
            "Affordance": list(self.affordances),
            "Scenario_name": self.scenario,
        }


def _json_object(response: Union[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    if isinstance(response, Mapping):
        return response
    if not isinstance(response, str):
        raise ContextParseError("Response must be a JSON string or mapping")
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ContextParseError("Malformed JSON code fence")
        if lines[0].strip().lower() not in {"```", "```json"}:
            raise ContextParseError("Only an optional JSON code fence is allowed")
        text = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ContextParseError(f"Invalid JSON: {exc.msg}") from exc
    if not isinstance(parsed, Mapping):
        raise ContextParseError("Top-level JSON value must be an object")
    return parsed


def _canonical_label(value: Any, allowed: Sequence[str], field: str) -> str:
    if not isinstance(value, str):
        raise ContextParseError(f"{field} labels must be strings")
    label = value.strip().upper().replace("-", "_").replace(" ", "_")
    if label not in allowed:
        raise ContextParseError(f"Unknown {field} label: {value!r}")
    return label


def _label_list(
    value: Any,
    allowed: Sequence[str],
    field: str,
    *,
    minimum: int,
    maximum: int,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ContextParseError(f"{field} must be a JSON array")
    if not minimum <= len(value) <= maximum:
        raise ContextParseError(
            f"{field} must contain between {minimum} and {maximum} labels"
        )
    labels = tuple(_canonical_label(item, allowed, field) for item in value)
    if len(set(labels)) != len(labels):
        raise ContextParseError(f"{field} contains duplicate labels")
    return labels


def parse_llm_response(response: Union[str, Mapping[str, Any]]) -> ParsedContext:
    """Parse one response without guessing missing fields or unknown labels."""

    obj = _json_object(response)
    keys = set(obj)
    if keys != _FIELDS:
        missing = sorted(_FIELDS - keys)
        extra = sorted(keys - _FIELDS)
        raise ContextParseError(f"JSON fields mismatch; missing={missing}, extra={extra}")

    situation = obj["Situation Understanding"]
    reasoning = obj["Reasoning"]
    if not isinstance(situation, str) or not situation.strip():
        raise ContextParseError("Situation Understanding must be a non-empty string")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ContextParseError("Reasoning must be a non-empty string")

    actions = _label_list(
        obj["Actions"], INTENTIONS, "Actions", minimum=1, maximum=3
    )
    affordances = _label_list(
        obj["Affordance"], AFFORDANCES, "Affordance", minimum=0, maximum=4
    )
    scenario = _canonical_label(obj["Scenario_name"], SCENARIOS, "Scenario_name")
    return ParsedContext(
        situation_understanding=situation.strip(),
        reasoning=reasoning.strip(),
        actions=actions,
        affordances=affordances,
        scenario=scenario,
    )


def encode_context(context: ParsedContext) -> np.ndarray:
    """Encode ranked actions, affordances, and scenario as ``float32[17]``.

    Action weights descend from ``N`` to one in the returned ranking.  The
    affordance and scenario sections use ordinary multi-hot encoding.
    """

    if not isinstance(context, ParsedContext):
        raise TypeError("encode_context expects ParsedContext from parse_llm_response")
    vector = np.zeros(CONTEXT_DIM, dtype=np.float32)
    action_index = {name: index for index, name in enumerate(INTENTIONS)}
    affordance_index = {name: index for index, name in enumerate(AFFORDANCES)}
    scenario_index = {name: index for index, name in enumerate(SCENARIOS)}

    count = len(context.actions)
    for rank, action in enumerate(context.actions):
        vector[action_index[action]] = float(count - rank)
    affordance_offset = len(INTENTIONS)
    for label in context.affordances:
        vector[affordance_offset + affordance_index[label]] = 1.0
    scenario_offset = len(INTENTIONS) + len(AFFORDANCES)
    vector[scenario_offset + scenario_index[context.scenario]] = 1.0
    return vector
