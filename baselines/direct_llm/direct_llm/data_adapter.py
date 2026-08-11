"""Leakage-safe adapter for Direct LLM post-crash trajectory prompts.

``ObservationSample`` deliberately has no future, lane-change phase, or
time-since-crossing member.  Prompt builders accept only that type.  Ground
truth can be loaded separately into ``TrajectorySample`` for offline metrics,
but prompt preparation never requests it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import scipy.io

from .config import DEFAULT_LAYOUT, DatasetLayout


class SampleValidationError(ValueError):
    """Raised when a MATLAB sample violates the documented array contract."""


@dataclass(frozen=True)
class ObservationSample:
    scenario_id: Any
    traj_id: Any
    x_hist: np.ndarray
    ego: np.ndarray
    neighbors: Mapping[str, np.ndarray]
    neighbor_masks: Mapping[str, np.ndarray]
    layout: DatasetLayout

    @property
    def current_frame(self) -> int:
        value = float(self.ego[-1, self.layout.ego.frame])
        if not np.isfinite(value):
            raise SampleValidationError("current ego frame must be finite")
        return int(round(value))

    @property
    def sample_key(self) -> str:
        return f"{_identifier_text(self.scenario_id)}:{_identifier_text(self.traj_id)}:{self.current_frame}"

    @property
    def sample_id(self) -> str:
        return self.sample_key

    @property
    def event_key(self) -> str:
        return _identifier_text(self.scenario_id)


@dataclass(frozen=True)
class TrajectorySample:
    observation: ObservationSample
    future: Optional[np.ndarray]

    @property
    def sample_id(self) -> str:
        return self.observation.sample_id


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
            raise SampleValidationError(f"missing required field: {name}")
        return container[name]
    if isinstance(container, np.void) and container.dtype.names and name in container.dtype.names:
        return container[name]
    if isinstance(container, np.ndarray) and container.dtype.names and name in container.dtype.names:
        return container[name]
    if hasattr(container, "_fieldnames") and name in container._fieldnames:
        return getattr(container, name)
    raise SampleValidationError(f"missing required field: {name}")


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


def _identifier_text(value: Any) -> str:
    value = _identifier(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(float(value)):
        number = float(value)
        return str(int(number)) if number.is_integer() else format(number, ".15g")
    text = str(value).strip()
    if not text:
        raise SampleValidationError("sample identifiers must be non-empty")
    return text


def _matrix(value: Any, name: str, minimum_columns: int) -> np.ndarray:
    value = _unwrap_scalar(value)
    try:
        matrix = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise SampleValidationError(f"{name} cannot be converted to a numeric matrix") from exc
    if matrix.ndim != 2:
        raise SampleValidationError(f"{name} must be two dimensional, got {matrix.shape}")
    if matrix.shape[1] < minimum_columns:
        raise SampleValidationError(
            f"{name} needs at least {minimum_columns} columns, got {matrix.shape[1]}"
        )
    return matrix


def _tail_or_pad(matrix: np.ndarray, steps: int, columns: int) -> np.ndarray:
    result = np.full((steps, columns), np.nan, dtype=np.float32)
    count = min(steps, matrix.shape[0])
    if count:
        result[-count:] = matrix[-count:, :columns]
    return result


def adapt_observation(
    raw_sample: Any,
    layout: DatasetLayout = DEFAULT_LAYOUT,
) -> ObservationSample:
    """Read observation fields only.

    In particular, this function never looks up ``y_future``, ``lane_status``,
    or ``time_since_crossing`` even if those fields exist in the MATLAB struct.
    """

    x_full = _matrix(_field(raw_sample, "x_hist"), "x_hist", layout.history.minimum_columns)
    if x_full.shape[0] < layout.history_steps:
        raise SampleValidationError(
            f"x_hist needs {layout.history_steps} observed rows, got {x_full.shape[0]}"
        )
    x_hist = np.array(
        x_full[-layout.history_steps :, : layout.history.minimum_columns],
        dtype=np.float32,
        copy=True,
    )
    if not np.isfinite(x_hist).all():
        raise SampleValidationError("x_hist contains non-finite observation values")

    ctx = _field(raw_sample, "ctx")
    ego_full = _matrix(_field(ctx, "ego"), "ctx.ego", layout.ego.minimum_columns)
    if ego_full.shape[0] < layout.history_steps:
        raise SampleValidationError(
            f"ctx.ego needs {layout.history_steps} observed rows, got {ego_full.shape[0]}"
        )
    ego = np.array(
        ego_full[-layout.history_steps :, : layout.ego.minimum_columns],
        dtype=np.float32,
        copy=True,
    )
    hc, ec = layout.history, layout.ego
    if not np.isfinite(ego[:, [ec.x, ec.y, ec.frame]]).all():
        raise SampleValidationError("ctx.ego contains invalid observed position or frame values")
    if not np.allclose(
        x_hist[:, [hc.x, hc.y]],
        ego[:, [ec.x, ec.y]],
        rtol=0.0,
        atol=layout.position_tolerance_m,
    ):
        raise SampleValidationError("x_hist and ctx.ego positions are not aligned")
    ego = np.nan_to_num(ego, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    neighbors: Dict[str, np.ndarray] = {}
    neighbor_masks: Dict[str, np.ndarray] = {}
    nc = layout.neighbor
    for source_field in layout.neighbor_roles:
        value = _optional_field(ctx, source_field)
        if value is None:
            padded = np.full(
                (layout.history_steps, nc.minimum_columns), np.nan, dtype=np.float32
            )
        else:
            matrix = _matrix(value, f"ctx.{source_field}", nc.minimum_columns)
            padded = _tail_or_pad(matrix, layout.history_steps, nc.minimum_columns)
        mask = (
            np.isfinite(padded[:, nc.vehicle_id])
            & (padded[:, nc.vehicle_id] > 0)
            & np.isfinite(padded[:, nc.x])
            & np.isfinite(padded[:, nc.y])
        )
        finite = np.nan_to_num(padded, nan=0.0, posinf=0.0, neginf=0.0)
        finite[~mask] = 0.0
        neighbors[source_field] = finite.astype(np.float32)
        neighbor_masks[source_field] = mask.astype(bool)

    return ObservationSample(
        scenario_id=_identifier(_field(raw_sample, "scenario_id")),
        traj_id=_identifier(_field(raw_sample, "traj_id")),
        x_hist=x_hist,
        ego=ego,
        neighbors=neighbors,
        neighbor_masks=neighbor_masks,
        layout=layout,
    )


def adapt_sample(
    raw_sample: Any,
    layout: DatasetLayout = DEFAULT_LAYOUT,
    *,
    include_future: bool = False,
) -> TrajectorySample:
    """Adapt a sample, loading ground truth only when explicitly requested."""

    observation = adapt_observation(raw_sample, layout)
    future: Optional[np.ndarray] = None
    if include_future:
        raw_future = _field(raw_sample, "y_future")
        matrix = _matrix(raw_future, "y_future", 2)
        if matrix.shape[0] < layout.future_steps:
            raise SampleValidationError(
                f"y_future needs {layout.future_steps} rows, got {matrix.shape[0]}"
            )
        future = np.array(matrix[: layout.future_steps, :2], dtype=np.float32, copy=True)
        if not np.isfinite(future).all():
            raise SampleValidationError("y_future contains non-finite labels")
    return TrajectorySample(observation=observation, future=future)


def target_frame(observation: ObservationSample) -> tuple[np.ndarray, float]:
    """Return the last target position and heading used by every conversion."""

    if not isinstance(observation, ObservationSample):
        raise TypeError("target_frame expects ObservationSample")
    hc = observation.layout.history
    origin = observation.x_hist[-1, [hc.x, hc.y]].astype(np.float64)
    heading = float(observation.x_hist[-1, hc.yaw])
    if not np.isfinite(heading):
        raise SampleValidationError("last observed target heading must be finite")
    # Local exports have used both radians and degrees; values outside a full
    # radian circle are unambiguously interpreted as degrees.
    if abs(heading) > 2.0 * np.pi + 1e-3:
        heading = float(np.deg2rad(heading))
    return origin.astype(np.float32), heading


def positions_to_local(points: np.ndarray, observation: ObservationSample) -> np.ndarray:
    """Convert global positions into last-target ``[forward,left]`` coordinates."""

    values = np.asarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("points must be a finite [N,2] matrix")
    origin, heading = target_frame(observation)
    delta = values - origin
    cosine, sine = float(np.cos(heading)), float(np.sin(heading))
    return np.stack(
        [
            cosine * delta[:, 0] + sine * delta[:, 1],
            -sine * delta[:, 0] + cosine * delta[:, 1],
        ],
        axis=1,
    ).astype(np.float32)


def vectors_to_local(vectors: np.ndarray, observation: ObservationSample) -> np.ndarray:
    """Rotate vectors into ``[forward,left]`` without translation."""

    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("vectors must be a finite [N,2] matrix")
    _, heading = target_frame(observation)
    cosine, sine = float(np.cos(heading)), float(np.sin(heading))
    return np.stack(
        [
            cosine * values[:, 0] + sine * values[:, 1],
            -sine * values[:, 0] + cosine * values[:, 1],
        ],
        axis=1,
    ).astype(np.float32)


def future_to_local(
    sample: TrajectorySample, *, expected_steps: Optional[int] = None
) -> np.ndarray:
    """Convert separately loaded ground truth for offline evaluation."""

    if not isinstance(sample, TrajectorySample):
        raise TypeError("future_to_local expects TrajectorySample")
    if sample.future is None:
        raise ValueError("sample has no future labels")
    steps = expected_steps or sample.observation.layout.future_steps
    if sample.future.shape != (steps, 2):
        raise ValueError(f"future must have shape ({steps},2), got {sample.future.shape}")
    return positions_to_local(sample.future, sample.observation)


def _load_raw_samples(path: Union[str, Path], key: Optional[str], limit: Optional[int]):
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    mat_path = Path(path)
    if not mat_path.is_file():
        raise FileNotFoundError(f"MATLAB data file not found: {mat_path}")
    payload = scipy.io.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    available = sorted(name for name in payload if not name.startswith("__"))
    if key is None:
        if len(available) != 1:
            raise KeyError(f"specify a MATLAB key; available keys are {available}")
        key = available[0]
    if key not in payload:
        raise KeyError(f"MATLAB key {key!r} not found; available keys are {available}")
    raw = payload[key]
    items = raw.reshape(-1) if isinstance(raw, np.ndarray) else [raw]
    return items if limit is None else items[:limit]


def load_mat_samples(
    path: Union[str, Path],
    key: Optional[str] = None,
    layout: DatasetLayout = DEFAULT_LAYOUT,
    *,
    include_future: bool = False,
    limit: Optional[int] = None,
) -> list[TrajectorySample]:
    """Load MATLAB samples, defaulting to observation-only access."""

    return [
        adapt_sample(item, layout=layout, include_future=include_future)
        for item in _load_raw_samples(path, key, limit)
    ]


def load_mat_observations(
    path: Union[str, Path],
    key: Optional[str] = None,
    layout: DatasetLayout = DEFAULT_LAYOUT,
    *,
    limit: Optional[int] = None,
) -> list[ObservationSample]:
    """Load only prompt-safe observations from a MATLAB split."""

    return [
        adapt_observation(item, layout=layout)
        for item in _load_raw_samples(path, key, limit)
    ]
