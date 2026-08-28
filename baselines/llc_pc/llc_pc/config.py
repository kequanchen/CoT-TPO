"""Configuration for the post-crash lane-change adaptation of LLC-PC.

All column indices are zero based.  The defaults match the MATLAB samples
used by CoT-TP; callers can provide another :class:`DatasetLayout` when a
locally prepared dataset uses a different ordering.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple, Union


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
    """Column layout of ``ctx.ego`` (shape ``[T, 19]``)."""

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
    """Column layout of each ``ctx.phys_*`` matrix (shape ``[T, 8]``)."""

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
    """Column layout of ``x_hist`` (shape ``[T, 6]``)."""

    x: int = 0
    y: int = 1
    longitudinal_speed: int = 2
    lateral_speed: int = 3
    acceleration: int = 4
    yaw: int = 5
    minimum_columns: int = 6


@dataclass(frozen=True)
class DatasetLayout:
    """Dataset and observation settings shared by the data-facing modules."""

    history_steps: int = 10
    history_hz: float = 10.0
    prediction_seconds: float = 5.0
    position_tolerance_m: float = 1e-3
    ego: EgoColumns = EgoColumns()
    neighbor: NeighborColumns = NeighborColumns()
    history: HistoryColumns = HistoryColumns()
    neighbor_roles: Tuple[str, ...] = NEIGHBOR_ROLES


@dataclass(frozen=True)
class RenderConfig:
    """Settings for the deterministic, dependency-free TC-map renderer."""

    width_px: int = 640
    height_px: int = 640
    adjacent_lanes_each_side: int = 1
    default_lane_width_m: float = 3.9
    longitudinal_behind_m: float = 35.0
    longitudinal_ahead_m: float = 55.0
    lateral_margin_m: float = 2.0
    highlighted_neighbors: int = 3


DEFAULT_LAYOUT = DatasetLayout()
DEFAULT_RENDER_CONFIG = RenderConfig()


_REQUIRED_SECTIONS = (
    "data",
    "paths",
    "llm",
    "context",
    "intention_points",
    "model",
    "train",
    "evaluation",
)


def load_config(path: Union[str, Path]) -> dict[str, Any]:
    """Load and minimally validate an LLC-PC JSON configuration.

    Relative data and artifact paths are resolved against the LLC-PC baseline
    directory, not against the caller's current working directory.  Public
    placeholders such as ``<PATH_TO_TRAIN_MAT>`` are intentionally left
    untouched so the checked-in example remains data-free.
    """

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"LLC-PC configuration not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("LLC-PC configuration must be a JSON object")
    missing = [name for name in _REQUIRED_SECTIONS if not isinstance(payload.get(name), Mapping)]
    if missing:
        raise ValueError(f"missing configuration object(s): {', '.join(missing)}")

    config = copy.deepcopy(dict(payload))
    _validate_config_values(config)
    baseline_root = Path(__file__).resolve().parents[1]
    for key in ("train_mat", "validation_mat", "test_mat"):
        config["data"][key] = _resolve_public_path(config["data"][key], baseline_root)
    for key, value in list(config["paths"].items()):
        config["paths"][key] = _resolve_public_path(value, baseline_root)
    return config


def dataset_layout_from_config(config: Mapping[str, Any]) -> DatasetLayout:
    """Construct the documented MATLAB layout from a loaded configuration."""

    data = config["data"]
    history_steps = int(data["history_steps"])
    sample_rate = float(data["sample_rate_hz"])
    future_steps = int(data["future_steps"])
    roles = tuple(str(item) for item in data.get("neighbor_roles", NEIGHBOR_ROLES))
    if len(roles) != 6 or len(set(roles)) != len(roles):
        raise ValueError("data.neighbor_roles must contain six unique roles")
    return DatasetLayout(
        history_steps=history_steps,
        history_hz=sample_rate,
        prediction_seconds=future_steps / sample_rate,
        neighbor_roles=roles,
    )


def load_configured_split(
    config: Mapping[str, Any],
    split: str,
    *,
    include_future: bool,
):
    """Load one configured split while keeping future access explicit.

    Prompt and retrieval code should always pass ``include_future=False``.
    Intention-point fitting, supervised training, validation, and final test
    evaluation are the expected callers that pass ``True``.
    """

    normalized = str(split).strip().lower()
    if normalized not in {"train", "validation", "test"}:
        raise ValueError("split must be 'train', 'validation', or 'test'")
    from .data_adapter import load_mat_samples

    path = str(config["data"][f"{normalized}_mat"])
    if _is_placeholder(path):
        raise ValueError(
            f"data.{normalized}_mat is still a public placeholder; "
            "copy the example configuration to a local file and set a private path"
        )
    key = str(config["data"][f"{normalized}_key"])
    return load_mat_samples(
        path,
        key=key,
        layout=dataset_layout_from_config(config),
        include_future=include_future,
    )


def _is_placeholder(value: str) -> bool:
    stripped = value.strip()
    return stripped.startswith("<") and stripped.endswith(">")


def _resolve_public_path(value: Any, root: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("configured paths must be non-empty strings")
    if _is_placeholder(value):
        return value
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return str(path.resolve())


def _validate_config_values(config: Mapping[str, Any]) -> None:
    data = config["data"]
    for split in ("train", "validation", "test"):
        for suffix in ("mat", "key"):
            field = f"{split}_{suffix}"
            value = data.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"data.{field} must be a non-empty string")
    if int(data.get("history_steps", 0)) <= 0:
        raise ValueError("data.history_steps must be positive")
    if int(data.get("future_steps", 0)) <= 0:
        raise ValueError("data.future_steps must be positive")
    if float(data.get("sample_rate_hz", 0.0)) <= 0:
        raise ValueError("data.sample_rate_hz must be positive")
    if int(config["context"].get("dimension", 0)) != 17:
        raise ValueError("context.dimension must be 17 for LLC-PC")
    if int(config["context"].get("k", 0)) <= 0:
        raise ValueError("context.k must be positive")
    if int(config["intention_points"].get("n_clusters", 0)) <= 0:
        raise ValueError("intention_points.n_clusters must be positive")
    if str(config["evaluation"].get("prediction_mode", "")).lower() != "top1":
        raise ValueError("evaluation.prediction_mode must be 'top1' for paper-aligned evaluation")
