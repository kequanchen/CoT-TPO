"""Construct and render a local straight-road TC-map from observed histories.

This renderer does not require an HD map.  It estimates parallel lane lines
from the ego vehicle's observed distances to its two lane boundaries and draws
only the one-second observation window.  PNG encoding uses NumPy and the Python
standard library, so no plotting dependency is required.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Union

import numpy as np

from .config import DEFAULT_RENDER_CONFIG, RenderConfig
from .data_adapter import ObservationSample


@dataclass(frozen=True)
class VehicleTrace:
    role: str
    local_xy: np.ndarray
    valid_mask: np.ndarray
    current_speed_mps: float
    current_distance_m: float
    highlight_rank: Optional[int] = None


@dataclass(frozen=True)
class TCMapScene:
    lane_boundaries_m: np.ndarray
    lane_width_m: float
    traces: Mapping[str, VehicleTrace]
    lateral_limits_m: tuple[float, float]
    longitudinal_limits_m: tuple[float, float]


def _require_observation(observation: ObservationSample) -> None:
    if not isinstance(observation, ObservationSample):
        raise TypeError(
            "TC-map functions accept ObservationSample only; pass sample.observation "
            "to keep future labels out of VLM inputs"
        )


def _local_points(x: np.ndarray, y: np.ndarray, ego_x: float, ego_y: float) -> np.ndarray:
    # Column 0 is lateral and column 1 is longitudinal.  The road longitudinal
    # axis is the dataset X direction; rendering maps positive longitudinal
    # displacement toward the top of the image.
    return np.stack([y - ego_y, x - ego_x], axis=1).astype(np.float32)


def build_tc_map_scene(
    observation: ObservationSample,
    render_config: RenderConfig = DEFAULT_RENDER_CONFIG,
) -> TCMapScene:
    """Build a deterministic scene using observation-period values only."""

    _require_observation(observation)
    layout = observation.layout
    hc, ec, nc = layout.history, layout.ego, layout.neighbor
    ego_x = float(observation.x_hist[-1, hc.x])
    ego_y = float(observation.x_hist[-1, hc.y])

    upper = observation.ego[:, ec.upper_boundary_distance]
    lower = observation.ego[:, ec.lower_boundary_distance]
    widths = upper + lower
    valid_widths = widths[np.isfinite(widths) & (widths > 1.5) & (widths < 8.0)]
    lane_width = (
        float(np.median(valid_widths))
        if valid_widths.size
        else render_config.default_lane_width_m
    )
    current_upper = float(upper[-1]) if np.isfinite(upper[-1]) and upper[-1] > 0 else lane_width / 2
    current_lower = float(lower[-1]) if np.isfinite(lower[-1]) and lower[-1] > 0 else lane_width / 2
    upper_line = -current_upper
    lower_line = current_lower
    lane_lines = [upper_line, lower_line]
    for offset in range(1, render_config.adjacent_lanes_each_side + 1):
        lane_lines.extend([upper_line - offset * lane_width, lower_line + offset * lane_width])
    lane_boundaries = np.asarray(sorted(lane_lines), dtype=np.float32)

    ego_local = _local_points(
        observation.x_hist[:, hc.x], observation.x_hist[:, hc.y], ego_x, ego_y
    )
    ego_speed = float(
        np.hypot(
            observation.x_hist[-1, hc.longitudinal_speed],
            observation.x_hist[-1, hc.lateral_speed],
        )
    )
    traces: dict[str, VehicleTrace] = {
        "ego": VehicleTrace(
            role="ego",
            local_xy=ego_local,
            valid_mask=observation.ego_mask.copy(),
            current_speed_mps=ego_speed,
            current_distance_m=0.0,
        )
    }

    distances: list[tuple[float, str]] = []
    for role in layout.neighbor_roles:
        matrix = observation.neighbors[role]
        mask = observation.neighbor_masks[role]
        local = _local_points(matrix[:, nc.x], matrix[:, nc.y], ego_x, ego_y)
        local[~mask] = 0.0
        if np.any(mask):
            last = int(np.flatnonzero(mask)[-1])
            distance = float(np.linalg.norm(local[last]))
            speed = float(
                np.hypot(matrix[last, nc.longitudinal_speed], matrix[last, nc.lateral_speed])
            )
            distances.append((distance, role))
        else:
            distance = float("inf")
            speed = 0.0
        traces[role] = VehicleTrace(
            role=role,
            local_xy=local,
            valid_mask=mask.copy(),
            current_speed_mps=speed,
            current_distance_m=distance,
        )

    nearest = {
        role: rank
        for rank, (_, role) in enumerate(
            sorted(distances)[: render_config.highlighted_neighbors]
        )
    }
    for role, rank in nearest.items():
        trace = traces[role]
        traces[role] = VehicleTrace(
            role=trace.role,
            local_xy=trace.local_xy,
            valid_mask=trace.valid_mask,
            current_speed_mps=trace.current_speed_mps,
            current_distance_m=trace.current_distance_m,
            highlight_rank=rank,
        )

    lateral_values = list(lane_boundaries)
    longitudinal_values = [
        -render_config.longitudinal_behind_m,
        render_config.longitudinal_ahead_m,
    ]
    for trace in traces.values():
        if np.any(trace.valid_mask):
            points = trace.local_xy[trace.valid_mask]
            lateral_values.extend(points[:, 0].tolist())
            longitudinal_values.extend(points[:, 1].tolist())
    lateral_min = min(lateral_values) - render_config.lateral_margin_m
    lateral_max = max(lateral_values) + render_config.lateral_margin_m
    longitudinal_min = min(longitudinal_values)
    longitudinal_max = max(longitudinal_values)
    return TCMapScene(
        lane_boundaries_m=lane_boundaries,
        lane_width_m=lane_width,
        traces=traces,
        lateral_limits_m=(float(lateral_min), float(lateral_max)),
        longitudinal_limits_m=(float(longitudinal_min), float(longitudinal_max)),
    )


def _draw_line(
    image: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    x0, y0 = start
    x1, y1 = end
    count = max(abs(x1 - x0), abs(y1 - y0), 1) + 1
    xs = np.rint(np.linspace(x0, x1, count)).astype(int)
    ys = np.rint(np.linspace(y0, y1, count)).astype(int)
    radius = max(0, thickness // 2)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            xx = xs + dx
            yy = ys + dy
            valid = (xx >= 0) & (xx < image.shape[1]) & (yy >= 0) & (yy < image.shape[0])
            image[yy[valid], xx[valid]] = color


def _png_bytes(image: np.ndarray) -> bytes:
    height, width, channels = image.shape
    if channels != 3 or image.dtype != np.uint8:
        raise ValueError("PNG encoder expects an HxWx3 uint8 array")

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    scanlines = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + chunk(b"IEND", b"")
    )


def render_tc_map_png(
    observation: ObservationSample,
    output_path: Optional[Union[str, Path]] = None,
    render_config: RenderConfig = DEFAULT_RENDER_CONFIG,
) -> bytes:
    """Render a TC-map PNG and optionally write it to ``output_path``."""

    scene = build_tc_map_scene(observation, render_config)
    width, height = render_config.width_px, render_config.height_px
    if width < 64 or height < 64:
        raise ValueError("TC-map dimensions must be at least 64 pixels")
    image = np.full((height, width, 3), (238, 240, 242), dtype=np.uint8)
    margin = 28
    lat_min, lat_max = scene.lateral_limits_m
    lon_min, lon_max = scene.longitudinal_limits_m

    def pixel(point: np.ndarray) -> tuple[int, int]:
        lateral, longitudinal = float(point[0]), float(point[1])
        px = margin + (lateral - lat_min) / max(lat_max - lat_min, 1e-6) * (width - 2 * margin)
        py = height - margin - (longitudinal - lon_min) / max(lon_max - lon_min, 1e-6) * (height - 2 * margin)
        return int(round(px)), int(round(py))

    # Lane lines are dashed to distinguish inferred local geometry from an HD map.
    for lateral in scene.lane_boundaries_m:
        segment = 5.0
        cursor = lon_min
        while cursor < lon_max:
            start = pixel(np.asarray([lateral, cursor], dtype=np.float32))
            end = pixel(np.asarray([lateral, min(cursor + segment, lon_max)], dtype=np.float32))
            _draw_line(image, start, end, (150, 155, 160), thickness=2)
            cursor += segment * 2

    colors = {
        "ego": (210, 45, 45),
        "other": (95, 102, 110),
        "highlight_0": (241, 138, 24),
        "highlight_1": (38, 155, 91),
        "highlight_2": (36, 105, 180),
    }
    for role, trace in scene.traces.items():
        points = trace.local_xy[trace.valid_mask]
        if not len(points):
            continue
        if role == "ego":
            color = colors["ego"]
        elif trace.highlight_rank is None:
            color = colors["other"]
        else:
            color = colors.get(f"highlight_{trace.highlight_rank}", colors["other"])
        pixels = [pixel(point) for point in points]
        for first, second in zip(pixels[:-1], pixels[1:]):
            _draw_line(image, first, second, color, thickness=3)
        current_x, current_y = pixels[-1]
        half = 5 if role == "ego" else 4
        x0, x1 = max(0, current_x - half), min(width, current_x + half + 1)
        y0, y1 = max(0, current_y - half), min(height, current_y + half + 1)
        image[y0:y1, x0:x1] = color

    encoded = _png_bytes(image)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
    return encoded
