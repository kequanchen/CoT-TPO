"""Configuration and MATLAB column layouts for the adapted LC-LLM baseline.

All array indices are zero based.  The public defaults match the private
post-crash lane-change samples used by CoT-TP, but no dataset path or data are
embedded in this package.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union


NEIGHBOR_ROLES: Tuple[str, ...] = (
    "phys_tl",
    "phys_tf",
    "phys_tff",
    "phys_ol",
    "phys_of",
    "phys_off",
)


@dataclass(frozen=True)
class EgoColumns:
    vehicle_id: int = 0
    lane_id: int = 1
    frame: int = 2
    x: int = 3
    y: int = 4
    upper_boundary_distance: int = 5
    lower_boundary_distance: int = 6
    lateral_speed: int = 15
    longitudinal_speed: int = 16
    acceleration: int = 17
    yaw: int = 18
    minimum_columns: int = 19


@dataclass(frozen=True)
class NeighborColumns:
    vehicle_id: int = 0
    lane_id: int = 1
    x: int = 2
    y: int = 3
    longitudinal_speed: int = 4
    lateral_speed: int = 5
    acceleration: int = 6
    yaw: int = 7
    minimum_columns: int = 8


@dataclass(frozen=True)
class HistoryColumns:
    x: int = 0
    y: int = 1
    longitudinal_speed: int = 2
    lateral_speed: int = 3
    acceleration: int = 4
    yaw: int = 5
    minimum_columns: int = 6


@dataclass(frozen=True)
class DatasetLayout:
    history_steps: int = 10
    future_steps: int = 50
    sample_rate_hz: float = 10.0
    position_tolerance_m: float = 1e-3
    ego: EgoColumns = EgoColumns()
    neighbor: NeighborColumns = NeighborColumns()
    history: HistoryColumns = HistoryColumns()
    neighbor_roles: Tuple[str, ...] = NEIGHBOR_ROLES

    @property
    def history_seconds(self) -> float:
        return self.history_steps / self.sample_rate_hz

    @property
    def prediction_seconds(self) -> float:
        return self.future_steps / self.sample_rate_hz


@dataclass(frozen=True)
class RoadConfig:
    """Road facts supplied by the user rather than inferred from lane IDs."""

    num_lanes: Optional[int] = None
    lane_id_to_position: tuple[tuple[int, str], ...] = ()
    neighbor_range_m: float = 200.0
    target_vehicle_type: str = "Vehicle"

    def lane_position(self, lane_id: int) -> Optional[str]:
        return dict(self.lane_id_to_position).get(int(lane_id))


@dataclass(frozen=True)
class PromptConfig:
    trajectory_precision: int = 2
    history_stride: int = 1


@dataclass(frozen=True)
class LabelConfig:
    """Training-label settings; these are never read by inference prompts.

    ``phase_left_event`` matches the released CoT-TP preprocessing, which keeps
    only left-lane-change events: anticipation/crossing phases (0/1) are left
    lane change and relaxation (2) is keep lane.  The more general
    ``future_lateral_displacement`` mode uses the final target-frame lateral
    displacement and ``lateral_threshold_m``.
    """

    intention_label_mode: str = "phase_left_event"
    lateral_threshold_m: float = 1.5


DEFAULT_LAYOUT = DatasetLayout()
DEFAULT_ROAD_CONFIG = RoadConfig()
DEFAULT_PROMPT_CONFIG = PromptConfig()
DEFAULT_LABEL_CONFIG = LabelConfig()


def load_config(path: Union[str, Path]) -> dict[str, Any]:
    """Load a JSON configuration and resolve local paths against this baseline.

    Angle-bracket placeholders are deliberately preserved so the checked-in
    example remains data-free.  The loader validates the shared data contract
    while leaving model-specific sections available to the training modules.
    """

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"LC-LLM configuration not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("LC-LLM configuration must be a JSON object")
    for section in ("data", "paths"):
        if not isinstance(payload.get(section), Mapping):
            raise ValueError(f"missing configuration object: {section}")

    config = copy.deepcopy(dict(payload))
    config.setdefault("road", {})
    config.setdefault("prompt", {})
    _validate_config(config)
    baseline_root = Path(__file__).resolve().parents[1]
    for key in ("train_mat", "test_mat"):
        config["data"][key] = _resolve_path(config["data"][key], baseline_root)
    for key, value in list(config["paths"].items()):
        config["paths"][key] = _resolve_path(value, baseline_root)
    return config


def dataset_layout_from_config(config: Mapping[str, Any]) -> DatasetLayout:
    data = config["data"]
    roles = tuple(str(role) for role in data.get("neighbor_roles", NEIGHBOR_ROLES))
    return DatasetLayout(
        history_steps=int(data.get("history_steps", 10)),
        future_steps=int(data.get("future_steps", 50)),
        sample_rate_hz=float(data.get("sample_rate_hz", 10.0)),
        neighbor_roles=roles,
    )


def road_config_from_config(config: Mapping[str, Any]) -> RoadConfig:
    road = config.get("road", {})
    raw_count = road.get("num_lanes")
    count = None if raw_count in (None, "", "unknown") else int(raw_count)
    raw_mapping = road.get("lane_id_to_position", {}) or {}
    if not isinstance(raw_mapping, Mapping):
        raise ValueError("road.lane_id_to_position must be an object")
    mapping: list[tuple[int, str]] = []
    for lane_id, position in raw_mapping.items():
        text = str(position).strip()
        if not text:
            raise ValueError("road lane positions must be non-empty strings")
        mapping.append((int(lane_id), text))
    return RoadConfig(
        num_lanes=count,
        lane_id_to_position=tuple(sorted(mapping)),
        neighbor_range_m=float(road.get("neighbor_range_m", 200.0)),
        target_vehicle_type=str(road.get("target_vehicle_type", "Vehicle")).strip()
        or "Vehicle",
    )


def prompt_config_from_config(config: Mapping[str, Any]) -> PromptConfig:
    prompt = config.get("prompt", {})
    return PromptConfig(
        trajectory_precision=int(prompt.get("trajectory_precision", 2)),
        history_stride=int(prompt.get("history_stride", 1)),
    )


def label_config_from_config(config: Mapping[str, Any]) -> LabelConfig:
    data = config["data"]
    return LabelConfig(
        intention_label_mode=str(
            data.get("intention_label_mode", "phase_left_event")
        ).strip(),
        lateral_threshold_m=float(data.get("intention_lateral_threshold_m", 1.5)),
    )


def load_configured_split(
    config: Mapping[str, Any],
    split: str,
    *,
    include_future: bool,
    limit: int | None = None,
):
    """Load one configured split with explicit control of future-label access."""

    normalized = str(split).strip().lower()
    if normalized not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    path = str(config["data"][f"{normalized}_mat"])
    if _is_placeholder(path):
        raise ValueError(
            f"data.{normalized}_mat is still a public placeholder; set it in a "
            "private local configuration"
        )
    key = str(config["data"][f"{normalized}_key"])
    from .data_adapter import load_mat_samples

    return load_mat_samples(
        path,
        key=key,
        layout=dataset_layout_from_config(config),
        include_future=include_future,
        limit=limit,
    )


def _validate_config(config: Mapping[str, Any]) -> None:
    data = config["data"]
    for key in ("train_mat", "test_mat", "train_key", "test_key"):
        if key not in data or not str(data[key]).strip():
            raise ValueError(f"data.{key} must be provided")
    layout = dataset_layout_from_config(config)
    if layout.history_steps <= 0 or layout.future_steps <= 0:
        raise ValueError("history_steps and future_steps must be positive")
    if layout.sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if len(layout.neighbor_roles) != 6 or len(set(layout.neighbor_roles)) != 6:
        raise ValueError("data.neighbor_roles must contain six unique roles")
    if layout.future_steps != 50 or abs(layout.sample_rate_hz - 10.0) > 1e-9:
        raise ValueError("the paper-comparison protocol requires 50 points at 10 Hz")
    road = road_config_from_config(config)
    if road.num_lanes is not None and road.num_lanes <= 0:
        raise ValueError("road.num_lanes must be positive or null")
    if road.neighbor_range_m <= 0:
        raise ValueError("road.neighbor_range_m must be positive")
    prompt = prompt_config_from_config(config)
    if prompt.trajectory_precision < 0 or prompt.trajectory_precision > 8:
        raise ValueError("prompt.trajectory_precision must be between 0 and 8")
    if prompt.history_stride <= 0:
        raise ValueError("prompt.history_stride must be positive")
    labels = label_config_from_config(config)
    if labels.intention_label_mode not in {
        "phase_left_event",
        "future_lateral_displacement",
    }:
        raise ValueError(
            "data.intention_label_mode must be 'phase_left_event' or "
            "'future_lateral_displacement'"
        )
    if labels.lateral_threshold_m <= 0:
        raise ValueError("data.intention_lateral_threshold_m must be positive")


def _is_placeholder(value: str) -> bool:
    stripped = value.strip()
    return stripped.startswith("<") and stripped.endswith(">")


def _resolve_path(value: Any, root: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("configured paths must be non-empty strings")
    if _is_placeholder(value):
        return value
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return str(path.resolve())
