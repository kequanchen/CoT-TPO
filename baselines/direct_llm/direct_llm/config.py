"""Configuration and MATLAB layouts for the adapted Direct LLM baseline.

The defaults describe the post-crash lane-change arrays used by CoT-TP.  No
private path, sample, model credential, or trajectory is embedded here.
"""

from __future__ import annotations

import copy
import json
import math
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
        # Ten samples at 10 Hz are the ten observation instants ending at t=0.
        return self.history_steps / self.sample_rate_hz

    @property
    def prediction_seconds(self) -> float:
        return self.future_steps / self.sample_rate_hz


@dataclass(frozen=True)
class PromptConfig:
    """Controls only observation-side prompt rendering."""

    context_mode: str = "target_and_neighbors"
    trajectory_precision: int = 2
    max_neighbors: int = 6


DEFAULT_LAYOUT = DatasetLayout()
DEFAULT_PROMPT_CONFIG = PromptConfig()


def load_config(path: Union[str, Path]) -> dict[str, Any]:
    """Load a private JSON config and resolve its local paths.

    Checked-in angle-bracket placeholders are deliberately left unresolved.
    Model and evaluation sections are retained for the other baseline modules.
    """

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Direct LLM configuration not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("Direct LLM configuration must be a JSON object")
    for section in ("data", "paths"):
        if not isinstance(payload.get(section), Mapping):
            raise ValueError(f"missing configuration object: {section}")

    config = copy.deepcopy(dict(payload))
    config.setdefault("prompt", {})
    _validate_config(config)
    baseline_root = Path(__file__).resolve().parents[1]
    config["data"]["test_mat"] = _resolve_path(
        config["data"]["test_mat"], baseline_root
    )
    for key, value in list(config["paths"].items()):
        config["paths"][key] = _resolve_path(value, baseline_root)
    return config


def dataset_layout_from_config(config: Mapping[str, Any]) -> DatasetLayout:
    data = config["data"]
    roles = tuple(str(item) for item in data.get("neighbor_roles", NEIGHBOR_ROLES))
    return DatasetLayout(
        history_steps=int(data.get("history_steps", 10)),
        future_steps=int(data.get("future_steps", 50)),
        sample_rate_hz=float(data.get("sample_rate_hz", 10.0)),
        neighbor_roles=roles,
    )


def prompt_config_from_config(config: Mapping[str, Any]) -> PromptConfig:
    prompt = config.get("prompt", {})
    return PromptConfig(
        context_mode=str(prompt.get("context_mode", "target_and_neighbors")).strip(),
        trajectory_precision=int(prompt.get("trajectory_precision", 2)),
        max_neighbors=int(prompt.get("max_neighbors", 6)),
    )


def load_configured_split(
    config: Mapping[str, Any],
    split: str = "test",
    *,
    include_future: bool = False,
    limit: Optional[int] = None,
):
    """Load the configured test split with explicit future-label access."""

    if str(split).strip().lower() != "test":
        raise ValueError("Direct LLM prompt preparation supports only the test split")
    configured = str(config["data"]["test_mat"])
    if _is_placeholder(configured):
        raise ValueError(
            "data.test_mat is still a public placeholder; set it in a private config"
        )
    from .data_adapter import load_mat_samples

    return load_mat_samples(
        configured,
        key=str(config["data"]["test_key"]),
        layout=dataset_layout_from_config(config),
        include_future=include_future,
        limit=limit,
    )


def _validate_config(config: Mapping[str, Any]) -> None:
    data = config["data"]
    for key in ("test_mat", "test_key"):
        if key not in data or not str(data[key]).strip():
            raise ValueError(f"data.{key} must be provided")
    if "test_prompts" not in config["paths"]:
        raise ValueError("paths.test_prompts must be provided")

    layout = dataset_layout_from_config(config)
    if layout.history_steps != 10:
        raise ValueError("the comparison protocol requires 10 observed points")
    if (
        layout.future_steps != 50
        or not math.isfinite(layout.sample_rate_hz)
        or abs(layout.sample_rate_hz - 10.0) > 1e-9
    ):
        raise ValueError("the comparison protocol requires 50 future points at 10 Hz")
    if len(layout.neighbor_roles) != 6 or len(set(layout.neighbor_roles)) != 6:
        raise ValueError("data.neighbor_roles must contain six unique source fields")

    prompt = prompt_config_from_config(config)
    if prompt.context_mode not in {"target_only", "target_and_neighbors"}:
        raise ValueError(
            "prompt.context_mode must be 'target_only' or 'target_and_neighbors'"
        )
    if not 0 <= prompt.trajectory_precision <= 8:
        raise ValueError("prompt.trajectory_precision must be between 0 and 8")
    if not 0 <= prompt.max_neighbors <= 6:
        raise ValueError("prompt.max_neighbors must be between 0 and 6")


def _is_placeholder(value: str) -> bool:
    text = value.strip()
    return text.startswith("<") and text.endswith(">")


def _resolve_path(value: Any, root: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("configured paths must be non-empty strings")
    if _is_placeholder(value):
        return value
    result = Path(value).expanduser()
    if not result.is_absolute():
        result = root / result
    return str(result.resolve())
