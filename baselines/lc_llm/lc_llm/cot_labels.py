"""Deterministic supervision-only labels for the adapted LC-LLM response.

The original LC-LLM paper programmatically labels CoT supervision using
driving rules.  These domain-adapted rules use the observed kinematics for
notable features and use a labelled train or validation sample to choose the supervised
intention and trajectory.  Nothing in this module is called by inference
prompt construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DEFAULT_LABEL_CONFIG, LabelConfig
from .data_adapter import (
    TrajectorySample,
    future_to_local,
    positions_to_local,
    vectors_to_local,
)


@dataclass(frozen=True)
class TrainingLabels:
    notable_features: tuple[str, ...]
    potential_behaviors: tuple[str, ...]
    intention: int
    future_local: np.ndarray


_ROLE_NAMES = {
    "phys_tl": "surrounding vehicle A",
    "phys_tf": "surrounding vehicle B",
    "phys_tff": "surrounding vehicle C",
    "phys_ol": "surrounding vehicle D",
    "phys_of": "surrounding vehicle E",
    "phys_off": "surrounding vehicle F",
}


def infer_training_intention(
    sample: TrajectorySample,
    config: LabelConfig = DEFAULT_LABEL_CONFIG,
) -> int:
    """Return a supervised class; this function requires training labels.

    The default reflects the private data preparation: only left-lane-change
    events are retained.  Status 0/1 denotes anticipation/crossing and status 2
    denotes the subsequent relaxation segment.  The optional displacement mode
    is suitable for a future dataset that includes all three classes.
    """

    if not isinstance(sample, TrajectorySample) or sample.future is None:
        raise ValueError("training intention requires a labelled TrajectorySample")
    mode = config.intention_label_mode
    if mode == "phase_left_event":
        status = sample.observation.lane_status
        if status in {0, 1}:
            return 1
        if status == 2:
            return 0
        raise ValueError(f"unexpected lane_status for left-event data: {status}")
    if mode == "future_lateral_displacement":
        lateral = float(future_to_local(sample)[-1, 1])
        if lateral > config.lateral_threshold_m:
            return 1
        if lateral < -config.lateral_threshold_m:
            return 2
        return 0
    raise ValueError(f"unknown intention_label_mode: {mode!r}")


def _observed_notable_features(sample: TrajectorySample) -> tuple[str, ...]:
    observation = sample.observation
    layout, hc, nc = observation.layout, observation.layout.history, observation.layout.neighbor
    velocity = vectors_to_local(
        observation.x_hist[-1:, [hc.longitudinal_speed, hc.lateral_speed]], observation
    )[0]
    acceleration = float(observation.x_hist[-1, hc.acceleration])
    features: list[str] = []
    lateral_kmh = float(velocity[1] * 3.6)
    if abs(lateral_kmh) > 1.5:
        direction = "left" if lateral_kmh > 0 else "right"
        features.append(
            f"the target has notable {direction} lateral motion ({abs(lateral_kmh):.2f} km/h)"
        )
    if acceleration > 0.5:
        features.append(f"the target is accelerating longitudinally ({acceleration:.2f} m/s^2)")
    elif acceleration < -0.5:
        features.append(f"the target is decelerating longitudinally ({acceleration:.2f} m/s^2)")

    for role in layout.neighbor_roles:
        mask = observation.neighbor_masks[role]
        if not np.any(mask):
            continue
        last = int(np.flatnonzero(mask)[-1])
        neighbor = observation.neighbors[role]
        relative = positions_to_local(
            neighbor[last : last + 1, [nc.x, nc.y]], observation
        )[0]
        if abs(float(relative[0])) > 100.0:
            continue
        neighbor_velocity = vectors_to_local(
            neighbor[last : last + 1, [nc.longitudinal_speed, nc.lateral_speed]],
            observation,
        )[0]
        speed_difference_kmh = float((neighbor_velocity[0] - velocity[0]) * 3.6)
        name = _ROLE_NAMES.get(role, role)
        if relative[0] >= 0 and speed_difference_kmh < -2.0:
            features.append(
                f"the {name} is ahead and {abs(speed_difference_kmh):.2f} km/h slower than the target"
            )
        elif abs(float(relative[0])) < 40.0:
            relation = "ahead" if relative[0] >= 0 else "behind"
            features.append(f"the {name} is observed {abs(float(relative[0])):.2f} m {relation}")
        if len(features) >= 4:
            break
    if not features:
        features.append("the target motion is approximately steady in the observed window")
    return tuple(features)


def _potential_behaviors(intention: int) -> tuple[str, ...]:
    if intention == 1:
        return ("change to the left lane while maintaining safe interaction gaps",)
    if intention == 2:
        return ("change to the right lane while maintaining safe interaction gaps",)
    if intention == 0:
        return ("follow the current lane and adjust speed to surrounding traffic",)
    raise ValueError("intention must be 0, 1, or 2")


def derive_training_labels(
    sample: TrajectorySample,
    config: LabelConfig = DEFAULT_LABEL_CONFIG,
) -> TrainingLabels:
    """Build joint Thought/Intention/Trajectory supervision for training only."""

    if not isinstance(sample, TrajectorySample) or sample.future is None:
        raise ValueError("derive_training_labels requires a labelled training sample")
    future = future_to_local(sample, expected_steps=sample.observation.layout.future_steps)
    intention = infer_training_intention(sample, config)
    return TrainingLabels(
        notable_features=_observed_notable_features(sample),
        potential_behaviors=_potential_behaviors(intention),
        intention=intention,
        future_local=future,
    )
