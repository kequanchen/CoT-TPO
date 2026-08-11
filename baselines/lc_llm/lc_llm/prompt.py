"""Paper-structured, domain-adapted LC-LLM prompts.

The role/coordinate/output structure is reconstructed from Figure 3 of:
M. Peng et al., "LC-LLM: Explainable lane-change intention and trajectory
predictions with Large Language Models," Communications in Transportation
Research 5 (2025) 100170, https://doi.org/10.1016/j.commtr.2025.100170.
The article is available under CC BY 4.0.  Dataset fields and sampling are
adapted here to one observed second, six post-crash vehicle roles, and 50
future points at 10 Hz.  This module never accesses future labels.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .config import (
    DEFAULT_LAYOUT,
    DEFAULT_PROMPT_CONFIG,
    DEFAULT_ROAD_CONFIG,
    DatasetLayout,
    PromptConfig,
    RoadConfig,
)
from .data_adapter import ObservationSample, positions_to_local, vectors_to_local


SOURCE_ATTRIBUTION = (
    "Prompt structure adapted from Peng et al. (2025), LC-LLM, Figure 3, "
    "CC BY 4.0, DOI:10.1016/j.commtr.2025.100170."
)

INTENTION_LABELS = {
    0: "Keep lane",
    1: "Left lane change",
    2: "Right lane change",
}

_ROLE_NAMES = {
    # Source-array role names are intentionally not exposed.  Their semantic
    # meaning changes across event phases, whereas the local coordinates below
    # remain valid and do not reveal the annotated lane-change phase.
    "phys_tl": "surrounding vehicle A",
    "phys_tf": "surrounding vehicle B",
    "phys_tff": "surrounding vehicle C",
    "phys_ol": "surrounding vehicle D",
    "phys_of": "surrounding vehicle E",
    "phys_off": "surrounding vehicle F",
}


def build_system_prompt(layout: DatasetLayout = DEFAULT_LAYOUT) -> str:
    """Return the Figure 3 system-message contract with adapted sampling."""

    if layout.future_steps != 50 or abs(layout.sample_rate_hz - 10.0) > 1e-9:
        raise ValueError("LC-LLM comparison requires 50 future points at 10 Hz")
    interval = 1.0 / layout.sample_rate_hz
    return f"""Role: You are an expert driving prediction model in an autonomous driving system. Predict the future driving intention and the next {layout.prediction_seconds:g} s trajectory of a target vehicle while accounting for interactions with nearby vehicles and roadway constraints.

Context:
- Coordinates: use a target-vehicle coordinate system centered at the target's current position (0, 0). The x-axis is parallel to the target's current heading. Positive y is to the target's left and negative y is to its right. Positions are measured in metres.

Output:
- Thought:
  - Notable features
  - Potential behaviors
- Final answer:
  - Intention: use exactly one of 0: Keep lane; 1: Left lane change; 2: Right lane change.
  - Trajectory (MOST IMPORTANT): output exactly {layout.future_steps} coordinate points, one every {interval:g} s, in chronological order.
  - Use this exact structure:
Thought:
- Notable feature: <observation-grounded feature>
- Potential behavior: <plausible behavior>
Final answer:
- Intention: \"<0, 1, or 2>: <label>\"
- Trajectory: \"[[x1, y1], [x2, y2], ..., [x{layout.future_steps}, y{layout.future_steps}]]\""""


def _number(value: float, precision: int) -> str:
    rendered = f"{float(value):.{precision}f}"
    return "0." + "0" * precision if rendered.startswith("-0.") and float(value) == 0 else rendered


def _points_text(points: np.ndarray, precision: int) -> str:
    return "[" + ", ".join(
        f"({_number(x, precision)}, {_number(y, precision)})" for x, y in points
    ) + "]"


def _history_indices(count: int, stride: int) -> list[int]:
    indices = list(range(0, count, stride))
    if not indices or indices[-1] != count - 1:
        indices.append(count - 1)
    return indices


def _road_description(observation: ObservationSample, road: RoadConfig) -> str:
    lane_id = int(round(float(observation.ego[-1, observation.layout.ego.lane_id])))
    position = road.lane_position(lane_id)
    if road.num_lanes is not None and position:
        return (
            f"The target vehicle is driving on a {road.num_lanes}-lane highway "
            f"and is located in the {position} lane."
        )
    if road.num_lanes is not None:
        return (
            f"The target vehicle is driving on a {road.num_lanes}-lane highway, "
            f"currently in observed lane ID {lane_id}. Its left/middle/right road "
            "position is not inferred from the lane ID."
        )
    return (
        "The target vehicle is driving on a multilane highway, currently in "
        f"observed lane ID {lane_id}. The total lane count and its left/middle/right "
        "road position are unknown."
    )


def _current_target_information(
    observation: ObservationSample,
    road: RoadConfig,
    prompt: PromptConfig,
) -> list[str]:
    hc = observation.layout.history
    history_global = observation.x_hist[:, [hc.x, hc.y]]
    history_local = positions_to_local(history_global, observation)
    indices = _history_indices(len(history_local), prompt.history_stride)
    velocity = vectors_to_local(
        observation.x_hist[-1:, [hc.longitudinal_speed, hc.lateral_speed]],
        observation,
    )[0]
    velocity_kmh = velocity * 3.6
    acceleration = float(observation.x_hist[-1, hc.acceleration])
    return [
        "The target vehicle information is as follows:",
        "- Velocity (km/h): "
        f"v_x = {_number(velocity_kmh[0], prompt.trajectory_precision)}, "
        f"v_y = {_number(velocity_kmh[1], prompt.trajectory_precision)}.",
        "- Longitudinal acceleration (m/s^2): "
        f"a_x = {_number(acceleration, prompt.trajectory_precision)}; "
        "lateral acceleration is unavailable.",
        f"- Type: {road.target_vehicle_type}; physical width and length are unavailable.",
        f"- Historical positions from the {observation.layout.history_seconds:g} s observation window "
        f"({len(indices)} points selected from {observation.layout.history_steps} frames at "
        f"{observation.layout.sample_rate_hz:g} Hz): "
        f"{_points_text(history_local[indices], prompt.trajectory_precision)}.",
    ]


def _relative_words(forward: float, left: float) -> str:
    longitudinal = "ahead" if forward >= 0 else "behind"
    lateral = "left" if left >= 0 else "right"
    return (
        f"{abs(forward):.2f} m {longitudinal} and "
        f"{abs(left):.2f} m to the {lateral}"
    )


def _surrounding_information(
    observation: ObservationSample,
    road: RoadConfig,
    prompt: PromptConfig,
) -> Iterable[str]:
    layout, nc = observation.layout, observation.layout.neighbor
    yield (
        "The observed surrounding vehicles within "
        f"{road.neighbor_range_m:g} m are listed as follows:"
    )
    for role in layout.neighbor_roles:
        name = _ROLE_NAMES.get(role, role)
        mask = observation.neighbor_masks[role]
        if not np.any(mask):
            yield f"- {name}: not observed in the input window."
            continue
        last = int(np.flatnonzero(mask)[-1])
        matrix = observation.neighbors[role]
        position = positions_to_local(matrix[last : last + 1, [nc.x, nc.y]], observation)[0]
        distance = float(np.linalg.norm(position))
        if distance > road.neighbor_range_m:
            yield f"- {name}: no vehicle within {road.neighbor_range_m:g} m."
            continue
        velocity = vectors_to_local(
            matrix[last : last + 1, [nc.longitudinal_speed, nc.lateral_speed]],
            observation,
        )[0] * 3.6
        yield (
            f"- {name}: {_relative_words(float(position[0]), float(position[1]))}; "
            "velocity (km/h) "
            f"v_x = {_number(velocity[0], prompt.trajectory_precision)}, "
            f"v_y = {_number(velocity[1], prompt.trajectory_precision)}."
        )


def build_user_prompt(
    observation: ObservationSample,
    road_config: RoadConfig = DEFAULT_ROAD_CONFIG,
    prompt_config: PromptConfig = DEFAULT_PROMPT_CONFIG,
) -> str:
    """Describe observation-period map, target, history, and six neighbours."""

    if not isinstance(observation, ObservationSample):
        raise TypeError(
            "build_user_prompt accepts ObservationSample only; pass sample.observation "
            "so future coordinates cannot enter the inference prompt"
        )
    lines = [_road_description(observation, road_config)]
    lines.extend(_current_target_information(observation, road_config, prompt_config))
    lines.extend(_surrounding_information(observation, road_config, prompt_config))
    return "\n".join(lines)


def format_llama2_prompt(system_prompt: str, user_prompt: str) -> str:
    """Format the single-turn Llama-2 chat prefix shown schematically in Fig. 5."""

    if not system_prompt.strip() or not user_prompt.strip():
        raise ValueError("system_prompt and user_prompt must be non-empty")
    return f"<s>[INST] <<SYS>>\n{system_prompt.strip()}\n<</SYS>>\n\n{user_prompt.strip()} [/INST]"


def build_prompt(
    observation: ObservationSample,
    road_config: RoadConfig = DEFAULT_ROAD_CONFIG,
    prompt_config: PromptConfig = DEFAULT_PROMPT_CONFIG,
) -> str:
    """Return a complete inference prefix containing observation data only."""

    if not isinstance(observation, ObservationSample):
        raise TypeError(
            "build_prompt accepts ObservationSample only; pass sample.observation "
            "so future coordinates cannot enter the inference prompt"
        )
    system = build_system_prompt(observation.layout)
    user = build_user_prompt(observation, road_config, prompt_config)
    return format_llama2_prompt(system, user)
