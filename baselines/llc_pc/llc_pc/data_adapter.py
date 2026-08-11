"""Read CoT-TP MATLAB samples and expose an observation-only representation.

The future trajectory is deliberately stored outside :class:`ObservationSample`.
Prompt and TC-map functions accept only ``ObservationSample`` instances, which
makes accidental use of future ground truth in VLM inputs harder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import scipy.io

from .config import DEFAULT_LAYOUT, DatasetLayout


class SampleValidationError(ValueError):
    """Raised when a MATLAB sample does not match the documented schema."""


@dataclass(frozen=True)
class ObservationSample:
    """Information available at prediction time.

    ``neighbors`` contains finite matrices; missing and invalid rows are set to
    zero and identified by the corresponding boolean arrays in
    ``neighbor_masks``.
    """

    scenario_id: Any
    traj_id: Any
    lane_status: int
    time_since_crossing: float
    x_hist: np.ndarray
    ego: np.ndarray
    neighbors: Mapping[str, np.ndarray]
    ego_mask: np.ndarray
    neighbor_masks: Mapping[str, np.ndarray]
    layout: DatasetLayout

    @property
    def sample_key(self) -> str:
        # A trajectory contains many sliding windows, so scenario+trajectory
        # alone is not unique.  The current frame makes the key stable within
        # a split; orchestration code additionally prefixes the split name.
        return f"{self.scenario_id}:{self.traj_id}:{self.current_frame}"

    @property
    def current_frame(self) -> int:
        return int(round(float(self.ego[-1, self.layout.ego.frame])))

    @property
    def event_key(self) -> str:
        """Event-level ID used to prevent same-crash KNN retrieval."""

        return str(self.scenario_id)


@dataclass(frozen=True)
class TrajectorySample:
    """One adapted sample with future labels separated from its observation."""

    observation: ObservationSample
    future: Optional[np.ndarray]


def _unwrap_scalar(value: Any) -> Any:
    while isinstance(value, np.ndarray) and (value.ndim == 0 or value.size == 1):
        value = value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _field(container: Any, name: str) -> Any:
    container = _unwrap_scalar(container)
    if isinstance(container, Mapping):
        if name not in container:
            raise SampleValidationError(f"Missing required field: {name}")
        return container[name]
    if isinstance(container, np.void) and container.dtype.names and name in container.dtype.names:
        return container[name]
    if isinstance(container, np.ndarray) and container.dtype.names and name in container.dtype.names:
        return container[name]
    if hasattr(container, "_fieldnames") and name in container._fieldnames:
        return getattr(container, name)
    raise SampleValidationError(f"Missing required field: {name}")


def _optional_field(container: Any, name: str, default: Any = None) -> Any:
    try:
        return _field(container, name)
    except SampleValidationError:
        return default


def _identifier(value: Any) -> Any:
    value = _unwrap_scalar(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"U", "S"}:
            return "".join(str(item) for item in value.reshape(-1))
        if value.size == 1:
            return _identifier(value.item())
    return value


def _number(value: Any, name: str, integer: bool = False) -> Union[int, float]:
    value = _unwrap_scalar(value)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SampleValidationError(f"{name} must be a scalar number") from exc
    if not np.isfinite(number):
        raise SampleValidationError(f"{name} must be finite")
    return int(number) if integer else number


def _matrix(value: Any, name: str, minimum_columns: int) -> np.ndarray:
    value = _unwrap_scalar(value)
    try:
        matrix = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise SampleValidationError(f"{name} cannot be converted to a numeric matrix") from exc
    if matrix.ndim != 2:
        raise SampleValidationError(f"{name} must be two dimensional, got shape {matrix.shape}")
    if matrix.shape[1] < minimum_columns:
        raise SampleValidationError(
            f"{name} requires at least {minimum_columns} columns, got {matrix.shape[1]}"
        )
    return matrix


def _tail_or_pad(matrix: np.ndarray, steps: int, columns: int) -> np.ndarray:
    """Right-align a history and pad missing early frames with NaN."""

    result = np.full((steps, columns), np.nan, dtype=np.float32)
    rows = min(steps, matrix.shape[0])
    if rows:
        result[-rows:] = matrix[-rows:, :columns]
    return result


def adapt_sample(
    raw_sample: Any,
    layout: DatasetLayout = DEFAULT_LAYOUT,
    *,
    include_future: bool = True,
) -> TrajectorySample:
    """Validate and adapt one MATLAB struct or equivalent Python mapping.

    Only the final ``layout.history_steps`` rows of historical arrays are used.
    ``y_future`` is never copied into the observation object.
    """

    if layout.history_steps <= 0:
        raise ValueError("history_steps must be positive")

    x_hist_full = _matrix(
        _field(raw_sample, "x_hist"), "x_hist", layout.history.minimum_columns
    )
    if x_hist_full.shape[0] < layout.history_steps:
        raise SampleValidationError(
            f"x_hist requires {layout.history_steps} observed rows, got {x_hist_full.shape[0]}"
        )
    x_hist = np.array(x_hist_full[-layout.history_steps :, : layout.history.minimum_columns], copy=True)
    if not np.isfinite(x_hist).all():
        raise SampleValidationError("x_hist contains non-finite observation values")

    ctx = _field(raw_sample, "ctx")
    ego_full = _matrix(_field(ctx, "ego"), "ctx.ego", layout.ego.minimum_columns)
    if ego_full.shape[0] < layout.history_steps:
        raise SampleValidationError(
            f"ctx.ego requires {layout.history_steps} observed rows, got {ego_full.shape[0]}"
        )
    ego = np.array(ego_full[-layout.history_steps :, : layout.ego.minimum_columns], copy=True)
    ego_position = ego[:, [layout.ego.x, layout.ego.y]]
    if not np.isfinite(ego_position).all():
        raise SampleValidationError("ctx.ego contains invalid observed positions")
    hist_position = x_hist[:, [layout.history.x, layout.history.y]]
    if not np.allclose(
        hist_position,
        ego_position,
        rtol=0.0,
        atol=layout.position_tolerance_m,
    ):
        raise SampleValidationError("x_hist and ctx.ego positions are not aligned")
    ego = np.nan_to_num(ego, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    ego_mask = np.ones(layout.history_steps, dtype=bool)

    neighbors: Dict[str, np.ndarray] = {}
    neighbor_masks: Dict[str, np.ndarray] = {}
    nc = layout.neighbor
    for role in layout.neighbor_roles:
        raw_neighbor = _optional_field(ctx, role)
        if raw_neighbor is None:
            padded = np.full(
                (layout.history_steps, nc.minimum_columns), np.nan, dtype=np.float32
            )
        else:
            neighbor_matrix = _matrix(raw_neighbor, f"ctx.{role}", nc.minimum_columns)
            padded = _tail_or_pad(
                neighbor_matrix, layout.history_steps, nc.minimum_columns
            )
        # MATLAB exports use NaN for most absent neighbors, while some legacy
        # exports use an all-zero row.  A positive finite vehicle ID separates
        # a valid vehicle at the coordinate origin from zero padding.
        mask = (
            np.isfinite(padded[:, nc.vehicle_id])
            & (padded[:, nc.vehicle_id] > 0)
            & np.isfinite(padded[:, nc.x])
            & np.isfinite(padded[:, nc.y])
        )
        padded = np.nan_to_num(padded, nan=0.0, posinf=0.0, neginf=0.0)
        padded[~mask] = 0.0
        neighbors[role] = padded.astype(np.float32)
        neighbor_masks[role] = mask.astype(bool)

    observation = ObservationSample(
        scenario_id=_identifier(_field(raw_sample, "scenario_id")),
        traj_id=_identifier(_field(raw_sample, "traj_id")),
        lane_status=int(_number(_field(raw_sample, "lane_status"), "lane_status", integer=True)),
        time_since_crossing=float(
            _number(_field(raw_sample, "time_since_crossing"), "time_since_crossing")
        ),
        x_hist=x_hist.astype(np.float32),
        ego=ego,
        neighbors=neighbors,
        ego_mask=ego_mask,
        neighbor_masks=neighbor_masks,
        layout=layout,
    )

    future: Optional[np.ndarray] = None
    if include_future:
        raw_future = _optional_field(raw_sample, "y_future")
        if raw_future is not None:
            future_matrix = _matrix(raw_future, "y_future", 2)
            future = np.array(future_matrix[:, :2], dtype=np.float32, copy=True)
            if not np.isfinite(future).all():
                raise SampleValidationError("y_future contains non-finite labels")

    return TrajectorySample(observation=observation, future=future)


def load_mat_samples(
    path: Union[str, Path],
    key: Optional[str] = None,
    layout: DatasetLayout = DEFAULT_LAYOUT,
    *,
    include_future: bool = True,
) -> list[TrajectorySample]:
    """Load and adapt all samples under one MATLAB variable.

    If ``key`` is omitted, the file must contain exactly one non-metadata
    variable.  No dataset paths or data files are bundled with this baseline.
    """

    mat_path = Path(path)
    if not mat_path.is_file():
        raise FileNotFoundError(f"MATLAB data file not found: {mat_path}")
    payload = scipy.io.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    available = sorted(name for name in payload if not name.startswith("__"))
    if key is None:
        if len(available) != 1:
            raise KeyError(f"Specify a MATLAB key; available keys are {available}")
        key = available[0]
    if key not in payload:
        raise KeyError(f"MATLAB key {key!r} not found; available keys are {available}")

    raw = payload[key]
    if isinstance(raw, np.ndarray):
        raw_samples = list(raw.reshape(-1))
    else:
        raw_samples = [raw]
    return [
        adapt_sample(item, layout=layout, include_future=include_future)
        for item in raw_samples
    ]
