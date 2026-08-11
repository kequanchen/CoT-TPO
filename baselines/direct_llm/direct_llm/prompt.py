"""Observation-only prompts for the adapted Direct LLM baseline.

The design follows the *methodological paradigm* of LMTraj-ZERO (Bae et al.,
CVPR 2024): serialize observed coordinates as language-model input and ask the
language model to emit future coordinates directly.  The wording and JSON
contract below are newly written for this repository; no upstream prompt code
is copied.  The adaptation produces one top-1, 5 s vehicle trajectory rather
than the original pedestrian benchmark's multimodal candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterable

import numpy as np

from .config import DEFAULT_LAYOUT, DEFAULT_PROMPT_CONFIG, DatasetLayout, PromptConfig
from .data_adapter import ObservationSample, positions_to_local


METHOD_ATTRIBUTION = (
    "Direct coordinate-generation paradigm adapted from LMTraj-ZERO: "
    "Bae, Lee, and Jeon, CVPR 2024. Prompt wording is original to this repository."
)


@dataclass(frozen=True)
class _RankedNeighbor:
    distance: float
    forward: float
    left: float
    source_index: int
    time_offsets: np.ndarray
    local_history: np.ndarray


def build_system_prompt(layout: DatasetLayout = DEFAULT_LAYOUT) -> str:
    """Specify direct coordinate extrapolation, with no reasoning interface."""

    if layout.history_steps != 10:
        raise ValueError("Direct LLM comparison requires 10 observed points")
    if layout.future_steps != 50 or abs(layout.sample_rate_hz - 10.0) > 1e-9:
        raise ValueError("Direct LLM comparison requires 50 future points at 10 Hz")
    interval = 1.0 / layout.sample_rate_hz
    return (
        "You predict a target vehicle's future coordinate sequence from observed "
        "coordinate sequences. Extrapolate the target motion directly. Return "
        "exactly one JSON object and nothing else. The object must have the single "
        f'key "future_trajectory", whose value contains exactly {layout.future_steps} '
        f"[x, y] points at {interval:g} s intervals in chronological order. Do not "
        "add any other fields or surrounding prose, Markdown, numbering, alternative "
        "trajectories, omitted points, formulas, or ellipses."
    )


def _rounded_points(points: np.ndarray, precision: int) -> list[list[float]]:
    values = np.asarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("prompt trajectories must be finite [N,2] matrices")
    result: list[list[float]] = []
    for forward, left in values:
        pair = [round(float(forward), precision), round(float(left), precision)]
        # Canonicalize negative zero for reproducible text and tests.
        result.append([0.0 if item == 0.0 else item for item in pair])
    return result


def _json_points(points: np.ndarray, precision: int) -> str:
    return json.dumps(
        _rounded_points(points, precision),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )


def _rank_observed_neighbors(observation: ObservationSample) -> list[_RankedNeighbor]:
    """Anonymize neighbors using only their last observed local geometry."""

    if not isinstance(observation, ObservationSample):
        raise TypeError("neighbor ranking expects ObservationSample")
    nc = observation.layout.neighbor
    ranked: list[_RankedNeighbor] = []
    for source_index, source_field in enumerate(observation.layout.neighbor_roles):
        mask = np.asarray(observation.neighbor_masks[source_field], dtype=bool)
        if not np.any(mask):
            continue
        indices = np.flatnonzero(mask)
        global_history = observation.neighbors[source_field][mask][:, [nc.x, nc.y]]
        local_history = positions_to_local(global_history, observation)
        forward, left = map(float, local_history[-1])
        ranked.append(
            _RankedNeighbor(
                distance=float(np.hypot(forward, left)),
                forward=forward,
                left=left,
                source_index=source_index,
                time_offsets=(
                    (indices - (observation.layout.history_steps - 1))
                    / observation.layout.sample_rate_hz
                ).astype(np.float32),
                local_history=local_history,
            )
        )
    # The private source-array role never determines the public identity except
    # as an unreachable exact-geometry tie breaker. A--F primarily mean nearest
    # to farthest at each neighbor's latest available observation.
    ranked.sort(
        key=lambda item: (
            round(item.distance, 6),
            round(item.forward, 6),
            round(item.left, 6),
            item.source_index,
        )
    )
    return ranked


def _neighbor_lines(
    observation: ObservationSample,
    prompt_config: PromptConfig,
) -> Iterable[str]:
    ranked = _rank_observed_neighbors(observation)[: prompt_config.max_neighbors]
    if not ranked:
        yield "No surrounding vehicle trajectory is available in the observation window."
        return
    yield (
        "Nearby vehicle trajectories are dynamically anonymized as Vehicle A, B, "
        "and so on by increasing distance from the target at each vehicle's latest "
        "available observation. Each sequence lists its observed time offsets from "
        "the target's t=0 and the corresponding positions:"
    )
    for offset, neighbor in enumerate(ranked):
        label = chr(ord("A") + offset)
        times = json.dumps(
            [round(float(value), 3) for value in neighbor.time_offsets],
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        yield (
            f"- Vehicle {label} time offsets (s): {times}; positions: "
            f"{_json_points(neighbor.local_history, prompt_config.trajectory_precision)}"
        )


def build_user_prompt(
    observation: ObservationSample,
    prompt_config: PromptConfig = DEFAULT_PROMPT_CONFIG,
) -> str:
    """Serialize only observed coordinates into a Direct LLM request."""

    if not isinstance(observation, ObservationSample):
        raise TypeError(
            "build_user_prompt accepts ObservationSample only; future labels and "
            "lane-change annotations are not prompt inputs"
        )
    if prompt_config.context_mode not in {"target_only", "target_and_neighbors"}:
        raise ValueError("unsupported context_mode")
    if not 0 <= prompt_config.trajectory_precision <= 8:
        raise ValueError("trajectory_precision must be between 0 and 8")
    if not 0 <= prompt_config.max_neighbors <= 6:
        raise ValueError("max_neighbors must be between 0 and 6")

    layout = observation.layout
    hc = layout.history
    target_local = positions_to_local(
        observation.x_hist[:, [hc.x, hc.y]], observation
    )
    interval = 1.0 / layout.sample_rate_hz
    lines = [
        "Coordinate convention: all positions are in metres in the target-centered "
        "frame at the final observed instant. x points forward along the target's "
        "final observed heading; y points left. The final target observation is [0,0].",
        f"Observation: {layout.history_steps} positions sampled every {interval:g} s "
        "and ordered from earliest to t=0.",
        "Target observed trajectory:",
        _json_points(target_local, prompt_config.trajectory_precision),
    ]
    if prompt_config.context_mode == "target_and_neighbors":
        lines.extend(_neighbor_lines(observation, prompt_config))
    else:
        lines.append("Use only the target trajectory above; no surrounding context is supplied.")
    lines.extend(
        [
            f"Predict the target's next {layout.prediction_seconds:g} s as exactly "
            f"{layout.future_steps} coordinate points at {interval:g} s intervals.",
            'Use the single top-level key "future_trajectory" and explicitly list '
            f"all {layout.future_steps} numeric [x, y] pairs in its JSON array.",
        ]
    )
    return "\n".join(lines)


def format_prompt_text(system_prompt: str, user_prompt: str) -> str:
    """Create a model-neutral audit string; inference retains both messages too."""

    system = str(system_prompt).strip()
    user = str(user_prompt).strip()
    if not system or not user:
        raise ValueError("system_prompt and user_prompt must be non-empty")
    return f"System:\n{system}\n\nUser:\n{user}\n\nAssistant:\n"


def build_prompt(
    observation: ObservationSample,
    prompt_config: PromptConfig = DEFAULT_PROMPT_CONFIG,
) -> str:
    """Return the model-neutral combined prompt for one observation."""

    if not isinstance(observation, ObservationSample):
        raise TypeError("build_prompt accepts ObservationSample only")
    return format_prompt_text(
        build_system_prompt(observation.layout),
        build_user_prompt(observation, prompt_config),
    )
