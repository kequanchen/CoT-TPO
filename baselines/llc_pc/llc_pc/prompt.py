"""Domain-adapted LLC-PC prompt built from observation-period information."""

from __future__ import annotations

from .data_adapter import ObservationSample
from .schema import AFFORDANCES, INTENTIONS, SCENARIOS
from .tc_map import build_tc_map_scene


_ROLE_NAMES = {
    "phys_tl": "target-lane leader",
    "phys_tf": "target-lane follower",
    "phys_tff": "second target-lane follower",
    "phys_ol": "original-lane leader",
    "phys_of": "original-lane follower",
    "phys_off": "second original-lane follower",
}
_HIGHLIGHT_COLORS = ("orange", "green", "blue")


def _labels(values: tuple[str, ...]) -> str:
    return ", ".join(values)


def _scene_caption(observation: ObservationSample) -> str:
    scene = build_tc_map_scene(observation)
    ego = scene.traces["ego"]
    lines = [
        f"The observed history contains {observation.layout.history_steps} frames at "
        f"{observation.layout.history_hz:g} Hz.",
        "The target vehicle is shown in red; unhighlighted surrounding vehicles "
        "are shown in gray.",
        f"The target vehicle's current speed is {ego.current_speed_mps:.2f} m/s.",
        f"The estimated local lane width is {scene.lane_width_m:.2f} m.",
    ]
    highlighted = sorted(
        (
            trace.highlight_rank,
            role,
            trace,
        )
        for role, trace in scene.traces.items()
        if trace.highlight_rank is not None
    )
    if not highlighted:
        lines.append("No surrounding vehicle has a valid observed position.")
    for rank, role, trace in highlighted:
        last = int(trace.valid_mask.nonzero()[0][-1])
        lateral, longitudinal = trace.local_xy[last]
        longitudinal_text = "ahead" if longitudinal >= 0 else "behind"
        lateral_text = "right" if lateral >= 0 else "left"
        lines.append(
            f"The {rank + 1} nearest vehicle is shown in {_HIGHLIGHT_COLORS[rank]}; "
            f"it is the {_ROLE_NAMES[role]}, {abs(float(longitudinal)):.2f} m "
            f"{longitudinal_text} and {abs(float(lateral)):.2f} m to the "
            f"{lateral_text}, traveling at {trace.current_speed_mps:.2f} m/s."
        )
    return "\n".join(lines)


def build_prompt(observation: ObservationSample) -> str:
    """Return the VLM prompt for one observation-only TC-map.

    The prompt retains the general intention/affordance/scenario interface of
    LLC-PC while describing the available post-crash straight-road evidence.
    It is newly written for this release and does not copy the upstream prompt.
    """

    if not isinstance(observation, ObservationSample):
        raise TypeError(
            "build_prompt expects ObservationSample; pass sample.observation so "
            "future coordinates cannot enter the prompt"
        )
    seconds = observation.layout.prediction_seconds
    caption = _scene_caption(observation)
    return f"""You are analyzing a post-crash lane-changing scene on a multilane road.

Use only the supplied TC-map and the observed one-second vehicle histories. Do not assume access to future coordinates or ground-truth outcomes. Infer the target vehicle's plausible motion over the next {seconds:g} seconds while accounting for the nearby vehicles and the locally inferred straight lane boundaries.

Observed scene summary:
{caption}

Return one to three ranked Actions, zero or more Affordance labels, and exactly one Scenario_name. The ranking in Actions must run from most likely to least likely.

Allowed Actions: {_labels(INTENTIONS)}
Allowed Affordance labels: {_labels(AFFORDANCES)}
Allowed Scenario_name labels: {_labels(SCENARIOS)}

Use STRAIGHT_LEFT or STRAIGHT_RIGHT for a lateral lane transition while continuing along the road. Reserve turn and U-turn labels for visible road geometry that genuinely supports those maneuvers. In this dataset, ON_STRAIGHT_ROAD will usually be appropriate, but select another scenario label when the observed evidence supports it.

Respond with one JSON object and no surrounding prose. Use exactly this schema; Actions and Affordance must be JSON arrays:
{{
  "Situation Understanding": "brief evidence-based scene description",
  "Reasoning": "brief explanation linking observed interactions to the labels",
  "Actions": ["STRAIGHT_LEFT", "STRAIGHT"],
  "Affordance": ["LEFT_ALLOW", "SLOW_ALLOW"],
  "Scenario_name": "ON_STRAIGHT_ROAD"
}}
"""
