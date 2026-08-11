"""Leakage-safe tensors for the self-contained LLC-PC predictor.

The observation tensor contains the target vehicle followed by the six roles
documented by CoT-TP.  Every spatial value is expressed in a target-centred
frame whose x axis points along the target vehicle's last observed heading.
Future labels are transformed only after all observation and retrieval
features have been constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

from .config import DEFAULT_RENDER_CONFIG, NEIGHBOR_ROLES
from .data_adapter import ObservationSample, TrajectorySample


AGENT_FEATURES = (
    "local_forward_m",
    "local_left_m",
    "forward_speed_mps",
    "left_speed_mps",
    "acceleration_mps2",
    "sin_relative_yaw",
    "cos_relative_yaw",
)


@dataclass(frozen=True)
class TensorizerConfig:
    history_steps: int = 10
    future_steps: int = 50
    lane_points: int = 20
    adjacent_lanes_each_side: int = 1
    default_lane_width_m: float = DEFAULT_RENDER_CONFIG.default_lane_width_m
    longitudinal_behind_m: float = DEFAULT_RENDER_CONFIG.longitudinal_behind_m
    longitudinal_ahead_m: float = DEFAULT_RENDER_CONFIG.longitudinal_ahead_m
    neighbor_roles: tuple[str, ...] = NEIGHBOR_ROLES

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "TensorizerConfig":
        data = config["data"]
        roles = tuple(str(role) for role in data.get("neighbor_roles", NEIGHBOR_ROLES))
        return cls(
            history_steps=int(data["history_steps"]),
            future_steps=int(data["future_steps"]),
            neighbor_roles=roles,
        )

    def __post_init__(self) -> None:
        if self.history_steps <= 0 or self.future_steps <= 0:
            raise ValueError("history_steps and future_steps must be positive")
        if self.lane_points < 2:
            raise ValueError("lane_points must be at least two")
        if self.adjacent_lanes_each_side < 0:
            raise ValueError("adjacent_lanes_each_side must be non-negative")
        if len(self.neighbor_roles) != 6 or len(set(self.neighbor_roles)) != 6:
            raise ValueError("LLC-PC requires six unique neighbour roles")


@dataclass(frozen=True)
class TensorizedSample:
    sample_id: str
    event_id: str
    agent_histories: np.ndarray
    agent_valid_mask: np.ndarray
    map_polylines: np.ndarray
    map_valid_mask: np.ndarray
    retrieval_features: np.ndarray
    future: Optional[np.ndarray]
    future_valid_mask: Optional[np.ndarray]


@dataclass(frozen=True)
class TensorizedBatch:
    sample_ids: np.ndarray
    event_ids: np.ndarray
    agent_histories: np.ndarray
    agent_valid_mask: np.ndarray
    map_polylines: np.ndarray
    map_valid_mask: np.ndarray
    retrieval_features: np.ndarray
    future: Optional[np.ndarray]
    future_valid_mask: Optional[np.ndarray]

    def __len__(self) -> int:
        return int(self.agent_histories.shape[0])


class FeatureStandardizer:
    """Serializable standardizer fitted only on training retrieval features."""

    FORMAT_VERSION = 1

    def __init__(self, epsilon: float = 1e-6) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = float(epsilon)
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.source_split: Optional[str] = None

    @property
    def is_fitted(self) -> bool:
        return self.mean_ is not None

    @property
    def feature_dim(self) -> int:
        self._require_fitted()
        assert self.mean_ is not None
        return int(self.mean_.shape[0])

    def fit(
        self, features: np.ndarray, *, source_split: str = "train"
    ) -> "FeatureStandardizer":
        if str(source_split).strip().lower() != "train":
            raise ValueError("retrieval standardizers may be fitted only on training data")
        values = _finite_matrix(features, "features")
        mean = values.mean(axis=0, dtype=np.float64)
        scale = values.std(axis=0, dtype=np.float64)
        scale[scale < self.epsilon] = 1.0
        self.mean_ = mean.astype(np.float32)
        self.scale_ = scale.astype(np.float32)
        self.source_split = "train"
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        self._require_fitted()
        values = _finite_matrix(features, "features")
        if values.shape[1] != self.feature_dim:
            raise ValueError(
                f"features have width {values.shape[1]}, expected {self.feature_dim}"
            )
        assert self.mean_ is not None and self.scale_ is not None
        return ((values - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(
        self, features: np.ndarray, *, source_split: str = "train"
    ) -> np.ndarray:
        return self.fit(features, source_split=source_split).transform(features)

    def save(self, path: Union[str, Path]) -> None:
        self._require_fitted()
        assert self.mean_ is not None and self.scale_ is not None
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            format_version=np.asarray(self.FORMAT_VERSION, dtype=np.int64),
            source_split=np.asarray("train"),
            epsilon=np.asarray(self.epsilon, dtype=np.float64),
            mean=self.mean_,
            scale=self.scale_,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FeatureStandardizer":
        with np.load(Path(path), allow_pickle=False) as data:
            if int(data["format_version"].item()) != cls.FORMAT_VERSION:
                raise ValueError("unsupported retrieval-standardizer format version")
            if str(data["source_split"].item()) != "train":
                raise ValueError("refusing to load a non-training standardizer")
            instance = cls(epsilon=float(data["epsilon"].item()))
            mean = np.asarray(data["mean"], dtype=np.float32)
            scale = np.asarray(data["scale"], dtype=np.float32)
            if (
                mean.ndim != 1
                or scale.shape != mean.shape
                or not np.isfinite(mean).all()
                or not np.isfinite(scale).all()
                or np.any(scale <= 0)
            ):
                raise ValueError("invalid retrieval-standardizer arrays")
            instance.mean_ = mean
            instance.scale_ = scale
            instance.source_split = "train"
        return instance

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("retrieval standardizer has not been fitted")


def tensorize_sample(
    sample: TrajectorySample,
    config: Optional[TensorizerConfig] = None,
) -> TensorizedSample:
    """Convert one adapted sample into fixed model and retrieval tensors."""

    if not isinstance(sample, TrajectorySample):
        raise TypeError("tensorize_sample expects a TrajectorySample")
    cfg = config or TensorizerConfig(history_steps=sample.observation.layout.history_steps)
    obs = sample.observation
    if obs.layout.history_steps != cfg.history_steps:
        raise ValueError(
            f"sample history has {obs.layout.history_steps} steps, expected {cfg.history_steps}"
        )
    if tuple(obs.layout.neighbor_roles) != tuple(cfg.neighbor_roles):
        raise ValueError("tensorizer neighbour-role order does not match the adapted sample")

    origin, heading = _target_frame(obs)
    histories, masks = _agent_tensors(obs, cfg, origin, heading)
    polylines, polyline_mask = _lane_polylines(obs, cfg)
    # This is deliberately finalized before labels are accessed below.
    retrieval = build_retrieval_features(histories, masks, polylines, polyline_mask)

    future = None
    future_mask = None
    if sample.future is not None:
        future, future_mask = _local_future(sample.future, cfg, origin, heading)

    return TensorizedSample(
        sample_id=obs.sample_key,
        event_id=str(obs.scenario_id),
        agent_histories=histories,
        agent_valid_mask=masks,
        map_polylines=polylines,
        map_valid_mask=polyline_mask,
        retrieval_features=retrieval,
        future=future,
        future_valid_mask=future_mask,
    )


def tensorize_samples(
    samples: Sequence[TrajectorySample],
    config: Optional[TensorizerConfig] = None,
) -> TensorizedBatch:
    """Stack samples into a batch with no framework-specific dependency."""

    if not samples:
        raise ValueError("samples must not be empty")
    cfg = config or TensorizerConfig(history_steps=samples[0].observation.layout.history_steps)
    tensors = [tensorize_sample(sample, cfg) for sample in samples]
    has_future = [item.future is not None for item in tensors]
    if any(has_future) and not all(has_future):
        raise ValueError("a tensor batch cannot mix labelled and unlabelled samples")
    return TensorizedBatch(
        sample_ids=np.asarray([item.sample_id for item in tensors], dtype=str),
        event_ids=np.asarray([item.event_id for item in tensors], dtype=str),
        agent_histories=np.stack([item.agent_histories for item in tensors]),
        agent_valid_mask=np.stack([item.agent_valid_mask for item in tensors]),
        map_polylines=np.stack([item.map_polylines for item in tensors]),
        map_valid_mask=np.stack([item.map_valid_mask for item in tensors]),
        retrieval_features=np.stack([item.retrieval_features for item in tensors]),
        future=(np.stack([item.future for item in tensors]) if all(has_future) else None),
        future_valid_mask=(
            np.stack([item.future_valid_mask for item in tensors]) if all(has_future) else None
        ),
    )


def build_retrieval_features(
    agent_histories: np.ndarray,
    agent_valid_mask: np.ndarray,
    map_polylines: np.ndarray,
    map_valid_mask: np.ndarray,
) -> np.ndarray:
    """Flatten deterministic observation tensors into one retrieval vector."""

    agents = np.asarray(agent_histories, dtype=np.float32)
    agent_mask = np.asarray(agent_valid_mask, dtype=bool)
    lanes = np.asarray(map_polylines, dtype=np.float32)
    lane_mask = np.asarray(map_valid_mask, dtype=bool)
    if agents.ndim != 3 or agents.shape[-1] != len(AGENT_FEATURES):
        raise ValueError("agent_histories must have shape [A, T, 7]")
    if agent_mask.shape != agents.shape[:2]:
        raise ValueError("agent_valid_mask must match agent_histories[:2]")
    if lanes.ndim != 3 or lanes.shape[-1] != 2:
        raise ValueError("map_polylines must have shape [P, L, 2]")
    if lane_mask.shape != lanes.shape[:2]:
        raise ValueError("map_valid_mask must match map_polylines[:2]")
    if not np.isfinite(agents).all() or not np.isfinite(lanes).all():
        raise ValueError("observation tensors must be finite")
    return np.concatenate(
        (
            agents.reshape(-1),
            agent_mask.astype(np.float32).reshape(-1),
            lanes.reshape(-1),
            lane_mask.astype(np.float32).reshape(-1),
        )
    ).astype(np.float32)


def standardizer_path_for_index(index_path: Union[str, Path]) -> Path:
    path = Path(index_path)
    return path.with_name(f"{path.stem}_standardizer.npz")


def _target_frame(observation: ObservationSample) -> tuple[np.ndarray, float]:
    hc = observation.layout.history
    origin = observation.x_hist[-1, [hc.x, hc.y]].astype(np.float32)
    heading = float(observation.x_hist[-1, hc.yaw])
    return origin, heading


def _rotate_xy(values: np.ndarray, heading: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    cosine, sine = float(np.cos(heading)), float(np.sin(heading))
    forward = cosine * values[..., 0] + sine * values[..., 1]
    left = -sine * values[..., 0] + cosine * values[..., 1]
    return np.stack((forward, left), axis=-1).astype(np.float32)


def _relative_yaw(yaw: np.ndarray, heading: float) -> np.ndarray:
    return (np.asarray(yaw, dtype=np.float32) - heading + np.pi) % (2.0 * np.pi) - np.pi


def _agent_tensors(
    observation: ObservationSample,
    config: TensorizerConfig,
    origin: np.ndarray,
    heading: float,
) -> tuple[np.ndarray, np.ndarray]:
    layout = observation.layout
    hc, nc = layout.history, layout.neighbor
    histories = np.zeros((7, config.history_steps, len(AGENT_FEATURES)), dtype=np.float32)
    masks = np.zeros((7, config.history_steps), dtype=bool)

    ego = observation.x_hist
    ego_position = _rotate_xy(ego[:, [hc.x, hc.y]] - origin, heading)
    ego_velocity = _rotate_xy(
        ego[:, [hc.longitudinal_speed, hc.lateral_speed]], heading
    )
    ego_yaw = _relative_yaw(ego[:, hc.yaw], heading)
    histories[0] = np.column_stack(
        (
            ego_position,
            ego_velocity,
            ego[:, hc.acceleration],
            np.sin(ego_yaw),
            np.cos(ego_yaw),
        )
    )
    masks[0] = observation.ego_mask

    for agent_index, role in enumerate(config.neighbor_roles, start=1):
        matrix = observation.neighbors[role]
        mask = observation.neighbor_masks[role]
        position = _rotate_xy(matrix[:, [nc.x, nc.y]] - origin, heading)
        velocity = _rotate_xy(
            matrix[:, [nc.longitudinal_speed, nc.lateral_speed]], heading
        )
        relative_yaw = _relative_yaw(matrix[:, nc.yaw], heading)
        features = np.column_stack(
            (
                position,
                velocity,
                matrix[:, nc.acceleration],
                np.sin(relative_yaw),
                np.cos(relative_yaw),
            )
        ).astype(np.float32)
        features[~mask] = 0.0
        histories[agent_index] = features
        masks[agent_index] = mask
    return histories, masks


def _lane_polylines(
    observation: ObservationSample, config: TensorizerConfig
) -> tuple[np.ndarray, np.ndarray]:
    ec = observation.layout.ego
    upper = observation.ego[:, ec.upper_boundary_distance]
    lower = observation.ego[:, ec.lower_boundary_distance]
    widths = upper + lower
    usable = widths[np.isfinite(widths) & (widths > 1.5) & (widths < 8.0)]
    lane_width = float(np.median(usable)) if usable.size else config.default_lane_width_m
    current_upper = float(upper[-1]) if np.isfinite(upper[-1]) and upper[-1] > 0 else lane_width / 2
    current_lower = float(lower[-1]) if np.isfinite(lower[-1]) and lower[-1] > 0 else lane_width / 2
    boundaries = [-current_upper, current_lower]
    for offset in range(1, config.adjacent_lanes_each_side + 1):
        boundaries.extend(
            [-current_upper - offset * lane_width, current_lower + offset * lane_width]
        )
    boundaries = sorted(boundaries)
    forward = np.linspace(
        -config.longitudinal_behind_m,
        config.longitudinal_ahead_m,
        config.lane_points,
        dtype=np.float32,
    )
    polylines = np.stack(
        [np.column_stack((forward, np.full_like(forward, lateral))) for lateral in boundaries]
    ).astype(np.float32)
    return polylines, np.ones(polylines.shape[:2], dtype=bool)


def _local_future(
    future_values: np.ndarray,
    config: TensorizerConfig,
    origin: np.ndarray,
    heading: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(future_values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("future must have shape [T, 2]")
    output = np.zeros((config.future_steps, 2), dtype=np.float32)
    mask = np.zeros(config.future_steps, dtype=bool)
    count = min(config.future_steps, values.shape[0])
    finite = np.isfinite(values[:count, :2]).all(axis=1)
    local = _rotate_xy(values[:count, :2] - origin, heading)
    valid_rows = np.flatnonzero(finite)
    output[valid_rows] = local[valid_rows]
    mask[:count] = finite
    return output, mask


def _finite_matrix(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty rank-2 array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array
