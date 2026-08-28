"""
Trajectory-to-text and LLM CoT reasoning for CoT-TP.

This script implements the paper-aligned LLM teacher step:
1. read one lane-changing sample from a MATLAB trajectory dataset;
2. convert the historical trajectory and surrounding-vehicle interactions into
   a structured English scene description;
3. ask an LLM to perform staged chain-of-thought driving-strategy reasoning;
4. save the prompt, the raw response, and the parsed JSON strategy labels.

The candidate-strategy vocabulary follows the 12-strategy schema described in
the manuscript appendix:
- anticipation/crossing: DECISIVE_MERGE, PROBING_APPROACH, ACCELERATE_PASS,
  DECELERATE_THEN_MERGE, MAINTAIN_AND_OBSERVE, YIELD_FOR_GAP2
- relaxation: ACCELERATE_STABILIZE, DECELERATE_STABILIZE, SPEED_MATCH,
  MAINTAIN_STABLE, HOLD_AND_WAIT, DECELERATE_AVOID_CRASH
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import scipy.io

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - only triggered in incomplete envs.
    OpenAI = None


MERGE_STRATEGIES = [
    "DECISIVE_MERGE",
    "PROBING_APPROACH",
    "ACCELERATE_PASS",
    "DECELERATE_THEN_MERGE",
    "MAINTAIN_AND_OBSERVE",
    "YIELD_FOR_GAP2",
]

RELAXATION_STRATEGIES = [
    "ACCELERATE_STABILIZE",
    "DECELERATE_STABILIZE",
    "SPEED_MATCH",
    "MAINTAIN_STABLE",
    "HOLD_AND_WAIT",
    "DECELERATE_AVOID_CRASH",
]


@dataclass(frozen=True)
class MatrixCols:
    """Column mapping for one trajectory matrix."""

    x: int
    y: int
    lon_v: int
    lat_v: int
    acc: int
    yaw: Optional[int] = None
    lane_u: Optional[int] = None


@dataclass
class VehicleBehavior:
    """Kinematic and semantic descriptors for one vehicle history."""

    lon_speed: float
    lat_speed: float
    accel: float
    current_x: float
    current_y: float
    yaw: float
    lane_u: Optional[float] = None
    lon_speed_start: float = 0.0
    lat_speed_start: float = 0.0
    accel_start: float = 0.0
    x_start: float = 0.0
    y_start: float = 0.0
    yaw_start: float = 0.0
    lane_u_start: Optional[float] = None
    driving_style: str = ""
    reaction_trend: str = ""
    yaw_desc: str = ""
    lane_pos_desc: str = ""
    lane_u_desc: str = ""


def calc_gap_clear(front_x: float, rear_x: float, vehicle_length: float) -> float:
    """
    Net longitudinal clearance.

    gap_clear = front_vehicle_x - rear_vehicle_x - vehicle_length
    Positive values indicate an available gap. Negative values indicate
    longitudinal overlap or a side-by-side condition.
    """

    return front_x - rear_x - vehicle_length


def load_mat_data(file_path: Path, data_key: str) -> np.ndarray:
    """Load trajectory samples from a MATLAB file."""

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    mat_data = scipy.io.loadmat(file_path)
    if data_key not in mat_data:
        available = sorted(k for k in mat_data.keys() if not k.startswith("__"))
        raise ValueError(f"MATLAB key '{data_key}' not found. Available keys: {available}")
    return mat_data[data_key].flatten()


def unpack_scalar(data: Any) -> Any:
    """Recursively unpack MATLAB/Numpy scalar wrappers."""

    if data is None:
        return None
    while isinstance(data, np.ndarray):
        if data.size == 0:
            return None
        if data.ndim == 0:
            return data.item()
        if data.size == 1:
            data = data.flat[0]
        else:
            break
    return data


def unpack_matrix(data: Any) -> Optional[np.ndarray]:
    """Recursively unpack MATLAB/Numpy matrix wrappers."""

    if data is None:
        return None
    result = data
    while isinstance(result, np.ndarray):
        if result.size == 0:
            return None
        if result.dtype == object and result.size == 1:
            result = result.flat[0]
        elif result.ndim == 0:
            result = result.item()
        else:
            break
    if isinstance(result, np.ndarray) and result.size > 0:
        return result
    return None


def parse_sample(sample: Any) -> Dict[str, Any]:
    """Parse one trajectory sample into ego and surrounding-vehicle matrices."""

    parsed = {"status": 0, "ego": None, "neighbors": {}}
    try:
        parsed["status"] = int(unpack_scalar(sample["lane_status"]))
    except Exception:
        pass

    try:
        ctx = unpack_scalar(sample["ctx"])
        if hasattr(ctx, "dtype") and ctx.dtype.names:
            if "ego" in ctx.dtype.names:
                parsed["ego"] = unpack_matrix(ctx["ego"])
            for key in ["phys_tl", "phys_tf", "phys_tff", "phys_ol", "phys_of", "phys_off"]:
                if key in ctx.dtype.names:
                    parsed["neighbors"][key] = unpack_matrix(ctx[key])
    except Exception:
        pass

    return parsed


def analyze_style(accel: float, lon_v: float) -> str:
    """Assign a compact driving-style descriptor for congested traffic."""

    if accel > 0.5:
        return "assertive acceleration"
    if accel < -0.5:
        return "cautious deceleration"
    if lon_v > 12:
        return "relatively fast"
    return "steady keeping"


def extract_behavior(data: Optional[np.ndarray], cols: MatrixCols, is_ego: bool = True) -> Optional[VehicleBehavior]:
    """Extract start/end kinematic descriptors from one vehicle history."""

    if not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[0] < 2:
        return None

    valid_data = data if is_ego else data[~np.isnan(data[:, cols.x])]
    if len(valid_data) < 2:
        return None

    start_frame = valid_data[0]
    end_frame = valid_data[-1]

    try:
        lon_v_end = float(end_frame[cols.lon_v])
        lat_v_end = float(end_frame[cols.lat_v])
        acc_end = float(end_frame[cols.acc])
        x_end = float(end_frame[cols.x])
        y_end = float(end_frame[cols.y])
        yaw_end = float(end_frame[cols.yaw]) if cols.yaw is not None else 0.0
        lon_v_start = float(start_frame[cols.lon_v])
        lat_v_start = float(start_frame[cols.lat_v])
        acc_start = float(start_frame[cols.acc])
        x_start = float(start_frame[cols.x])
        y_start = float(start_frame[cols.y])
        yaw_start = float(start_frame[cols.yaw]) if cols.yaw is not None else 0.0
    except IndexError:
        return None

    lane_u_end = None
    lane_u_start = None
    lane_u_desc = ""
    lane_pos_desc = ""

    if is_ego and cols.lane_u is not None:
        try:
            lane_u_end = float(end_frame[cols.lane_u])
            lane_u_start = float(start_frame[cols.lane_u])
            if lane_u_end < 0.5:
                lane_u_desc = f"LANE_U={lane_u_end:.3f}; very close to the target-lane boundary"
                lane_pos_desc = "lateral position is close to the lane boundary; strong merge intent"
            elif lane_u_end < 1.0:
                lane_u_desc = f"LANE_U={lane_u_end:.3f}; moving toward the target-lane boundary"
                lane_pos_desc = "approaching the lane boundary; clear lane-change preparation"
            else:
                lane_u_desc = f"LANE_U={lane_u_end:.3f}; still near the lane center"
                lane_pos_desc = "still near the lane center; merge intent is weak"
        except Exception:
            lane_u_end = None
            lane_u_start = None

    if not lane_pos_desc:
        if lat_v_end > 0.2:
            lane_pos_desc = "moving toward the target lane"
        elif lat_v_end < -0.2:
            lane_pos_desc = "moving away from the target lane"
        else:
            lane_pos_desc = "laterally stable"

    if yaw_end > 0.03:
        yaw_desc = "heading toward the target lane"
    elif yaw_end < -0.03:
        yaw_desc = "heading away from the target lane"
    else:
        yaw_desc = "heading is nearly aligned"

    acc_change = acc_end - acc_start
    if acc_change > 0.3:
        reaction = "acceleration tendency is increasing"
    elif acc_change < -0.3:
        reaction = "deceleration tendency is increasing"
    else:
        reaction = "power output is stable"

    return VehicleBehavior(
        lon_speed=lon_v_end,
        lat_speed=lat_v_end,
        accel=acc_end,
        current_x=x_end,
        current_y=y_end,
        yaw=yaw_end,
        lane_u=lane_u_end,
        lon_speed_start=lon_v_start,
        lat_speed_start=lat_v_start,
        accel_start=acc_start,
        x_start=x_start,
        y_start=y_start,
        yaw_start=yaw_start,
        lane_u_start=lane_u_start,
        driving_style=analyze_style(acc_end, lon_v_end),
        reaction_trend=reaction,
        yaw_desc=yaw_desc,
        lane_pos_desc=lane_pos_desc,
        lane_u_desc=lane_u_desc,
    )


def compute_game_metrics(
    ego: VehicleBehavior,
    neighbor: Optional[VehicleBehavior],
    role: str,
    vehicle_length: float,
    vehicle_width: float,
) -> Optional[Dict[str, Any]]:
    """Compute ego-neighbor interaction descriptors."""

    if neighbor is None:
        return None

    if neighbor.current_x > ego.current_x:
        gap_clear = calc_gap_clear(neighbor.current_x, ego.current_x, vehicle_length)
        position = "ahead"
        position_desc = f"ahead of ego with {gap_clear:.2f} m net clearance"
    else:
        gap_clear = calc_gap_clear(ego.current_x, neighbor.current_x, vehicle_length)
        position = "behind"
        position_desc = f"behind ego with {gap_clear:.2f} m net clearance"

    dy = abs(neighbor.current_y - ego.current_y)
    dv = neighbor.lon_speed - ego.lon_speed

    ttc = 99.0
    if position == "behind" and dv > 0.01:
        ttc = (gap_clear + vehicle_length) / dv
    elif position == "ahead" and dv < -0.01:
        ttc = (gap_clear + vehicle_length) / abs(dv)

    side_by_side = gap_clear < 0
    risk = "low"
    if side_by_side and dy < (vehicle_width + 0.5):
        risk = "critical side-by-side risk"
    elif ttc < 2.0 or dy < vehicle_width:
        risk = "high"
    elif ttc < 4.0 or dy < vehicle_width + 0.8:
        risk = "medium"
    if position == "ahead" and role == "phys_tl" and gap_clear < -(vehicle_length * 0.5):
        risk = "high overlap with the target-lane leader"

    return {
        "role": role,
        "gap_clear": float(gap_clear),
        "dy": float(dy),
        "dv": float(dv),
        "ttc": float(ttc),
        "position": position,
        "position_desc": position_desc,
        "side_by_side": side_by_side,
        "risk_level": risk,
    }


def compute_pair_metrics(
    rear_veh: Optional[VehicleBehavior],
    front_veh: Optional[VehicleBehavior],
    role: str,
    vehicle_length: float,
) -> Optional[Dict[str, Any]]:
    """Compute interaction descriptors between two surrounding vehicles."""

    if rear_veh is None or front_veh is None:
        return None
    if front_veh.current_x < rear_veh.current_x:
        rear_veh, front_veh = front_veh, rear_veh

    gap_clear = calc_gap_clear(front_veh.current_x, rear_veh.current_x, vehicle_length)
    dv = rear_veh.lon_speed - front_veh.lon_speed
    ttc = gap_clear / dv if dv > 0.01 and gap_clear > 0 else 99.0

    return {
        "role": role,
        "gap_clear": float(gap_clear),
        "dv": float(dv),
        "ttc": float(ttc),
        "trend": "closing" if dv > 0.3 else "opening" if dv < -0.3 else "stable",
    }


def compute_gap_history(
    ego: Optional[VehicleBehavior],
    neighbor: Optional[VehicleBehavior],
    vehicle_length: float,
) -> Optional[Dict[str, Any]]:
    """Compute one-second change in ego-neighbor net clearance."""

    if ego is None or neighbor is None:
        return None
    if neighbor.current_x > ego.current_x:
        start_gap = calc_gap_clear(neighbor.x_start, ego.x_start, vehicle_length)
        end_gap = calc_gap_clear(neighbor.current_x, ego.current_x, vehicle_length)
    else:
        start_gap = calc_gap_clear(ego.x_start, neighbor.x_start, vehicle_length)
        end_gap = calc_gap_clear(ego.current_x, neighbor.current_x, vehicle_length)

    return {
        "gap_clear_start": float(start_gap),
        "gap_clear_end": float(end_gap),
        "gap_change": float(end_gap - start_gap),
        "gap_trend": "opening" if end_gap > start_gap + 0.3 else "closing" if end_gap < start_gap - 0.3 else "stable",
    }


def compute_gap_history_pair(
    rear_veh: Optional[VehicleBehavior],
    front_veh: Optional[VehicleBehavior],
    vehicle_length: float,
) -> Optional[Dict[str, Any]]:
    """Compute one-second change in net clearance between two non-ego vehicles."""

    if rear_veh is None or front_veh is None:
        return None
    if front_veh.current_x < rear_veh.current_x:
        return None

    start_gap = calc_gap_clear(front_veh.x_start, rear_veh.x_start, vehicle_length)
    end_gap = calc_gap_clear(front_veh.current_x, rear_veh.current_x, vehicle_length)
    return {
        "gap_clear_start": float(start_gap),
        "gap_clear_end": float(end_gap),
        "gap_change": float(end_gap - start_gap),
        "gap_trend": "opening" if end_gap > start_gap + 0.3 else "closing" if end_gap < start_gap - 0.3 else "stable",
    }


def speed_trend(current: float, previous: float) -> str:
    if current > previous:
        return "increasing"
    if current < previous:
        return "decreasing"
    return "stable"


def generate_relaxation_prompt(
    sample: Any,
    vehicle_length: float,
    vehicle_width: float,
    ego_cols: MatrixCols,
    neighbor_cols: MatrixCols,
) -> Optional[str]:
    """Generate the relaxation-phase CoT prompt."""

    parsed = parse_sample(sample)
    ego = extract_behavior(parsed["ego"], ego_cols, is_ego=True)
    if ego is None:
        return None

    neighbors: Dict[str, Dict[str, Any]] = {}
    for key in ["phys_tl", "phys_tf", "phys_tff", "phys_ol", "phys_of", "phys_off"]:
        n_data = parsed["neighbors"].get(key)
        n_state = extract_behavior(n_data, neighbor_cols, is_ego=False)
        if n_state is not None:
            neighbors[key] = {
                "state": n_state,
                "metrics": compute_game_metrics(ego, n_state, key, vehicle_length, vehicle_width),
            }

    ol = neighbors.get("phys_ol")
    of_ = neighbors.get("phys_of")
    tl_side = neighbors.get("phys_tl")
    tf_side = neighbors.get("phys_tf")

    safe_following_dist = ego.lon_speed * 1.5
    lat_moving = abs(ego.lat_speed) > 0.1
    yaw_not_straight = abs(ego.yaw) > 0.03

    prompt = f"""
Task: post-merge relaxation strategy reasoning for a lane-changing vehicle.

Phase: RELAXATION
The ego vehicle has entered the target lane. The goal is no longer to force an
insertion, but to stabilize after the lane change.

Reasoning objectives:
1. Evaluate car-following and speed matching with OL, the current-lane leader.
2. Evaluate rear-end safety with OF, the current-lane follower.
3. Check whether lateral velocity and yaw are converging to a stable lane-keeping state.
4. Use TL/TF only as optional residual side-lane interaction checks.

Physical constraints:
- Vehicle length: {vehicle_length:.1f} m.
- Vehicle width: {vehicle_width:.1f} m.
- Recommended following distance: current speed x 1.5 s = {safe_following_dist:.1f} m.

Ego state:
- Longitudinal speed: {ego.lon_speed:.2f} m/s.
- Acceleration: {ego.accel:.2f} m/s^2.
- Lateral speed: {ego.lat_speed:.3f} m/s ({'still laterally moving' if lat_moving else 'laterally stable'}).
- Yaw: {ego.yaw:.3f} rad ({'not fully aligned' if yaw_not_straight else 'nearly aligned'}).
- Driving style: {ego.driving_style}.

Current-lane leader OL:
"""

    if ol:
        ol_s = ol["state"]
        ol_m = ol["metrics"]
        ol_gap = ol_m["gap_clear"]
        following_status = "safe" if ol_gap > safe_following_dist else "close" if ol_gap > safe_following_dist * 0.5 else "too close"
        prompt += f"""
- OL speed: {ol_s.lon_speed:.2f} m/s; acceleration: {ol_s.accel:.2f} m/s^2.
- Net clearance to OL: {ol_gap:.2f} m ({following_status}; recommended > {safe_following_dist:.1f} m).
- Relative speed: ego is {'faster' if ego.lon_speed > ol_s.lon_speed else 'slower'} by {abs(ego.lon_speed - ol_s.lon_speed):.2f} m/s.
- Lateral distance: {ol_m['dy']:.2f} m.
- Risk level: {ol_m['risk_level']}.
"""
    else:
        prompt += "- OL is absent. Downstream space is open; stabilization and smooth acceleration are feasible.\n"

    prompt += "\nCurrent-lane follower OF:\n"
    if of_:
        of_s = of_["state"]
        of_m = of_["metrics"]
        prompt += f"""
- OF speed: {of_s.lon_speed:.2f} m/s; acceleration: {of_s.accel:.2f} m/s^2.
- Net clearance from OF to ego: {of_m['gap_clear']:.2f} m.
- Relative speed dv = OF - ego: {of_m['dv']:.2f} m/s ({'OF is closing in' if of_m['dv'] > 0.3 else 'OF is not closing in'}).
- TTC: {of_m['ttc']:.1f} s.
- Lateral distance: {of_m['dy']:.2f} m.
- Risk level: {of_m['risk_level']}.
"""
    else:
        prompt += "- OF is absent. Rear-end pressure is low.\n"

    prompt += "\nResidual side-lane checks:\n"
    if tl_side:
        tl_s = tl_side["state"]
        tl_m = tl_side["metrics"]
        prompt += f"- TL: gap_clear={tl_m['gap_clear']:.2f} m, dy={tl_m['dy']:.2f} m, speed={tl_s.lon_speed:.2f} m/s, risk={tl_m['risk_level']}.\n"
    else:
        prompt += "- TL is absent.\n"
    if tf_side:
        tf_s = tf_side["state"]
        tf_m = tf_side["metrics"]
        prompt += f"- TF: gap_clear={tf_m['gap_clear']:.2f} m, dy={tf_m['dy']:.2f} m, speed={tf_s.lon_speed:.2f} m/s, risk={tf_m['risk_level']}.\n"
    else:
        prompt += "- TF is absent.\n"

    prompt += f"""
Structured analysis protocol:
1. Stabilization check: determine whether lateral speed and yaw have converged.
2. Following assessment: if OL exists, decide whether to decelerate, speed-match, or accelerate.
3. Rear-safety check: if OF exists, avoid abrupt deceleration when rear-end risk is high.
4. Residual side-lane check: if TL/TF still creates a close side-by-side risk, prioritize HOLD_AND_WAIT.
5. Select one primary strategy from this relaxation strategy pool:
   {', '.join(RELAXATION_STRATEGIES)}

Decision hints:
- Close side-lane residual interaction -> HOLD_AND_WAIT.
- Severe front-leader conflict or urgent front clearance issue -> DECELERATE_AVOID_CRASH.
- OL is close but not critical -> DECELERATE_STABILIZE.
- Speed should converge to OL or surrounding flow -> SPEED_MATCH.
- Downstream space is sufficient after merging -> ACCELERATE_STABILIZE.
- Already stable with no strong pressure -> MAINTAIN_STABLE.

Return only a valid JSON object in the following schema:
{{
  "phase": "RELAXATION",
  "reasoning": {{
    "stabilization_check": "text",
    "following_assessment": "text",
    "rear_safety": "text",
    "side_lane_check": "text",
    "decision_logic": "text"
  }},
  "decision": {{
    "primary_strategy": "ACCELERATE_STABILIZE / DECELERATE_STABILIZE / SPEED_MATCH / MAINTAIN_STABLE / HOLD_AND_WAIT / DECELERATE_AVOID_CRASH",
    "strategy_scores": {{
      "ACCELERATE_STABILIZE": 0.0,
      "DECELERATE_STABILIZE": 0.0,
      "SPEED_MATCH": 0.0,
      "MAINTAIN_STABLE": 0.0,
      "HOLD_AND_WAIT": 0.0,
      "DECELERATE_AVOID_CRASH": 0.0
    }},
    "lateral_intent": 0.0,
    "longitudinal_intent": 0.0,
    "stability_level": 0.0,
    "confidence": 0.0
  }},
  "prediction_guidance": {{
    "expected_lateral_displacement_3s": "numeric value in meters or short text",
    "expected_speed_change": "ACCEL / KEEP / DECEL",
    "target_speed": "numeric value in m/s or N/A",
    "key_interaction": "OL / OF / TL / TF / None",
    "time_to_full_stability": "1-5 s"
  }}
}}
"""
    return prompt


def generate_cot_prompt(
    sample: Any,
    vehicle_length: float,
    vehicle_width: float,
    ego_cols: MatrixCols,
    neighbor_cols: MatrixCols,
) -> Optional[str]:
    """Generate the CoT prompt for anticipation, crossing, or relaxation."""

    parsed = parse_sample(sample)
    status = parsed.get("status", 0)
    if status == 2:
        return generate_relaxation_prompt(sample, vehicle_length, vehicle_width, ego_cols, neighbor_cols)

    ego = extract_behavior(parsed["ego"], ego_cols, is_ego=True)
    if ego is None:
        return None

    phase_code = "ANTICIPATION" if status == 0 else "CROSSING"
    phase_name = "Anticipation" if status == 0 else "Crossing"
    include_progress = status == 0

    neighbors: Dict[str, Dict[str, Any]] = {}
    for key in ["phys_tl", "phys_tf", "phys_tff"]:
        n_data = parsed["neighbors"].get(key)
        n_state = extract_behavior(n_data, neighbor_cols, is_ego=False)
        if n_state is not None:
            neighbors[key] = {
                "state": n_state,
                "metrics": compute_game_metrics(ego, n_state, key, vehicle_length, vehicle_width),
            }

    tf = neighbors.get("phys_tf")
    tl = neighbors.get("phys_tl")
    tff = neighbors.get("phys_tff")
    tf_gap_hist = compute_gap_history(ego, tf["state"], vehicle_length) if tf else None
    tl_gap_hist = compute_gap_history(ego, tl["state"], vehicle_length) if tl else None

    gap1_info = None
    gap1_current = None
    if tl and tf:
        tl_x = tl["state"].current_x
        tf_x = tf["state"].current_x
        if tl_x > tf_x:
            gap1_current = calc_gap_clear(tl_x, tf_x, vehicle_length)
            gap1_info = compute_gap_history_pair(tf["state"], tl["state"], vehicle_length)

    gap2_info = None
    gap2_current = None
    tf_tff_pressure = None
    if tf and tff:
        tf_x = tf["state"].current_x
        tff_x = tff["state"].current_x
        if tf_x > tff_x:
            gap2_current = calc_gap_clear(tf_x, tff_x, vehicle_length)
            gap2_info = compute_gap_history_pair(tff["state"], tf["state"], vehicle_length)
            tf_tff_pressure = compute_pair_metrics(tff["state"], tf["state"], "TFF follows TF", vehicle_length)

    ego_tf_position = "unknown"
    tf_alongside_ego = False
    tf_ahead_of_ego = False
    ego_tf_gap_clear = 0.0
    if tf:
        tf_m = tf["metrics"]
        ego_tf_gap_clear = tf_m["gap_clear"]
        if ego_tf_gap_clear < -1.0:
            tf_alongside_ego = True
            ego_tf_position = "side-by-side with TF"
        elif tf_m["position"] == "ahead":
            tf_ahead_of_ego = True
            ego_tf_position = "TF ahead of ego"
        else:
            ego_tf_position = "TF behind ego"

    ego_tl_position = "unknown"
    ego_tl_gap_clear = 0.0
    ego_has_position_advantage = False
    if tl:
        tl_m = tl["metrics"]
        ego_tl_gap_clear = tl_m["gap_clear"]
        if ego_tl_gap_clear < -1.0:
            ego_tl_position = "side-by-side with TL"
            ego_has_position_advantage = True
        elif tl_m["position"] == "ahead":
            ego_tl_position = "TL ahead of ego"
        else:
            ego_tl_position = "ego ahead of TL"
            ego_has_position_advantage = True

    lane_change_progress = None
    lane_change_progress_desc = ""
    if include_progress:
        if ego.lane_u is not None:
            if ego.lane_u <= 0.3:
                lane_change_progress = 0.9
                lane_change_progress_desc = "most of the vehicle has entered the target lane"
            elif ego.lane_u <= 0.5:
                lane_change_progress = 0.7
                lane_change_progress_desc = "about half of the vehicle has entered the target lane"
            elif ego.lane_u <= 0.8:
                lane_change_progress = 0.4
                lane_change_progress_desc = "the vehicle has started to enter the target lane"
            elif ego.lane_u <= 1.2:
                lane_change_progress = 0.2
                lane_change_progress_desc = "approaching the lane boundary"
            else:
                lane_change_progress = 0.1
                lane_change_progress_desc = "still near the original-lane center"
        else:
            lane_change_progress = 0.3 if ego.lat_speed > 0.3 else 0.1
            lane_change_progress_desc = "estimated from lateral speed because LANE_U is unavailable"

    ego_table = f"""
Ego one-second history:
- Longitudinal speed: {ego.lon_speed_start:.2f} -> {ego.lon_speed:.2f} m/s ({speed_trend(ego.lon_speed, ego.lon_speed_start)}).
- Lateral speed: {ego.lat_speed_start:.3f} -> {ego.lat_speed:.3f} m/s ({ego.lane_pos_desc}).
- Acceleration: {ego.accel_start:.2f} -> {ego.accel:.2f} m/s^2 ({ego.reaction_trend}).
- Yaw: {ego.yaw_start:.3f} -> {ego.yaw:.3f} rad ({ego.yaw_desc}).
"""
    if include_progress:
        ego_table += f"- Lane-boundary distance LANE_U: {ego.lane_u_start if ego.lane_u_start is not None else 'N/A'} -> {ego.lane_u if ego.lane_u is not None else 'N/A'} ({ego.lane_u_desc or 'not available'}).\n"

    prompt = f"""
Task: lane-changing game-strategy reasoning for a post-crash congested freeway scene.

Phase: {phase_code} ({phase_name})
{'The ego vehicle is still in the original lane and is preparing to change lanes.' if status == 0 else 'The ego vehicle is crossing the lane boundary; do not use LANE_U as a progress cue.'}

Role and objective:
You are an experienced lane-changing driver in congested post-crash traffic.
The downstream crash requires a left lane change. Based on the past 1 second of
observed trajectories, infer the most plausible strategy over the next 3 seconds.

Core multi-gap logic:
1. Evaluate whether Gap1, the gap between TL and TF, is usable.
2. If Gap1 is not usable, evaluate whether the ego should yield and use Gap2, the gap between TF and TFF.
3. The ego-TF relative position is the key factor that determines whether Gap1 or Gap2 is more plausible.

Physical constraints:
- Vehicle length: {vehicle_length:.1f} m. gap_clear = front_x - rear_x - vehicle_length.
- Vehicle width: {vehicle_width:.1f} m. Lateral distance below the vehicle width indicates side-contact risk.

Scene perception:
{ego_table}
"""

    if include_progress:
        prompt += f"""
Lane-change progress cue:
- LANE_U: {ego.lane_u if ego.lane_u is not None else 'N/A'}.
- Estimated progress: {lane_change_progress:.0%}.
- Interpretation: {lane_change_progress_desc}.
This cue should affect how assertive the ego vehicle can be when the gap is otherwise feasible.
"""

    prompt += "\nRelative position with TL (target-lane leader):\n"
    if tl:
        prompt += f"- Ego-TL relation: {ego_tl_position}; net clearance={ego_tl_gap_clear:.2f} m.\n"
        prompt += f"- TL speed={tl['state'].lon_speed:.2f} m/s; ego speed={ego.lon_speed:.2f} m/s; TL risk={tl['metrics']['risk_level']}.\n"
        if tl_gap_hist:
            prompt += f"- Ego-TL gap history: {tl_gap_hist['gap_clear_start']:.2f} -> {tl_gap_hist['gap_clear_end']:.2f} m ({tl_gap_hist['gap_trend']}).\n"
        prompt += f"- Positional advantage: {'yes' if ego_has_position_advantage else 'no'}.\n"
    else:
        prompt += "- TL is absent; the front of the target lane is open.\n"

    prompt += "\nRelative position with TF (target-lane follower):\n"
    if tf:
        tf_m = tf["metrics"]
        tf_s = tf["state"]
        ego_minus_tf_speed = ego.lon_speed - tf_s.lon_speed
        prompt += f"- Ego-TF relation: {ego_tf_position}; net clearance={ego_tf_gap_clear:.2f} m.\n"
        prompt += f"- TF speed={tf_s.lon_speed:.2f} m/s; ego is {'faster' if ego_minus_tf_speed > 0 else 'slower'} by {abs(ego_minus_tf_speed):.2f} m/s.\n"
        prompt += f"- dv(TF-ego)={tf_m['dv']:.2f} m/s; TTC={tf_m['ttc']:.1f} s; dy={tf_m['dy']:.2f} m; risk={tf_m['risk_level']}.\n"
        prompt += f"- TF reaction trend: {tf_s.reaction_trend}; driving style: {tf_s.driving_style}.\n"
        if tf_gap_hist:
            prompt += f"- Ego-TF gap history: {tf_gap_hist['gap_clear_start']:.2f} -> {tf_gap_hist['gap_clear_end']:.2f} m ({tf_gap_hist['gap_trend']}).\n"
    else:
        prompt += "- TF is absent; rear-side resistance to the lane change is low.\n"

    gap1_size = f"{gap1_current:.2f} m" if gap1_current is not None else "N/A"
    gap1_trend = gap1_info["gap_trend"] if gap1_info else "N/A"
    gap2_size = f"{gap2_current:.2f} m" if gap2_current is not None else "N/A"
    gap2_trend = gap2_info["gap_trend"] if gap2_info else "N/A"
    if tf_alongside_ego or tf_ahead_of_ego:
        gap1_reachable = "requires changing the ego-TF relative position first"
    elif tf:
        gap1_reachable = "directly reachable if TF is not closing aggressively"
    else:
        gap1_reachable = "no TF blockage"

    prompt += f"""
Gap comparison:
- Gap1 (TL-TF): current net clearance={gap1_size}; trend={gap1_trend}; reachability={gap1_reachable}.
- Gap2 (TF-TFF): current net clearance={gap2_size}; trend={gap2_trend}; use condition=yield to TF or let TF pull away first.
"""
    if tf_tff_pressure:
        prompt += f"- TFF-TF pressure: dv(TFF-TF)={tf_tff_pressure['dv']:.2f} m/s; TTC={tf_tff_pressure['ttc']:.1f} s; trend={tf_tff_pressure['trend']}.\n"
    if tff:
        prompt += f"- TFF state: speed={tff['state'].lon_speed:.2f} m/s; acceleration={tff['state'].accel:.2f} m/s^2; risk to ego={tff['metrics']['risk_level']}.\n"

    prompt += f"""
Structured analysis protocol:
1. Positional advantage: evaluate whether ego already occupies part of Gap1.
2. Ego-TL interaction: determine whether the leader supports or constrains a Gap1 merge.
3. Ego-TF interaction: infer whether TF is yielding, neutral, or blocking.
4. Gap comparison: decide whether Gap1 or Gap2 is more feasible.
5. Select one primary strategy from this merge-strategy pool:
   {', '.join(MERGE_STRATEGIES)}

Decision hints:
- Strong positional advantage and feasible Gap1 -> DECISIVE_MERGE.
- Uncertain but feasible gap -> PROBING_APPROACH.
- Ego can pass TF or the relevant target-lane vehicle safely -> ACCELERATE_PASS.
- Deceleration is needed before merging safely -> DECELERATE_THEN_MERGE.
- Conditions are unclear or side risk is high -> MAINTAIN_AND_OBSERVE.
- Gap1 is unsafe, blocked, or already missed -> YIELD_FOR_GAP2.

Return only a valid JSON object in the following schema:
{{
  "phase": "{phase_code}",
  "reasoning": {{
    "position_advantage": "text",
    "ego_tl_analysis": "text",
    "ego_tf_analysis": "text",
    "gap_comparison": "text",
    "decision_logic": "text"
  }},
  "decision": {{
    "primary_strategy": "DECISIVE_MERGE / PROBING_APPROACH / ACCELERATE_PASS / DECELERATE_THEN_MERGE / MAINTAIN_AND_OBSERVE / YIELD_FOR_GAP2",
    "target_gap": "GAP1 / GAP2 / EITHER",
    "strategy_scores": {{
      "DECISIVE_MERGE": 0.0,
      "PROBING_APPROACH": 0.0,
      "ACCELERATE_PASS": 0.0,
      "DECELERATE_THEN_MERGE": 0.0,
      "MAINTAIN_AND_OBSERVE": 0.0,
      "YIELD_FOR_GAP2": 0.0
    }},
    "aggressiveness": 0.0,
    "risk_tolerance": 0.0,
    "lateral_intent": 0.0,
    "longitudinal_intent": 0.0,
    "confidence": 0.0
  }},
  "prediction_guidance": {{
    "expected_lateral_displacement_3s": "numeric value in meters or short text",
    "expected_speed_change": "ACCEL / KEEP / DECEL",
    "key_interaction": "TF / TL / TFF / None",
    "gap_switch_probability": 0.0,
    "lane_change_completion_probability": 0.0
  }}
}}
"""
    return prompt


def make_client(api_key_env: str, base_url: Optional[str]) -> Any:
    """Create an OpenAI-compatible client from an environment variable."""

    if OpenAI is None:
        raise ImportError("The openai package is required. Install it with `pip install openai`.")
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {api_key_env} is not set.")
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs)


def call_llm(
    prompt: str,
    client: Any,
    model: str,
    temperature: float,
    max_tokens: int,
    enable_thinking: bool = False,
) -> Dict[str, Any]:
    """Call the LLM and parse a JSON object from the response."""

    kwargs: Dict[str, Any] = {}
    if enable_thinking:
        kwargs["extra_body"] = {"enable_thinking": True}

    content = ""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert autonomous-driving decision analyst. "
                        "Reason over the traffic scene and return only the required JSON object. "
                        "Respect the role definitions: TL is the target-lane leader, TF is the "
                        "target-lane follower, and TFF is the follower behind TF."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        print("served_model =", getattr(response, "model", None))
        print("usage =", getattr(response, "usage", None))
        content = response.choices[0].message.content

        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        if json_start == -1 or json_end <= json_start:
            return {"success": False, "raw_response": content, "error": "No JSON object found in LLM response."}

        parsed = json.loads(content[json_start:json_end])
        return {"success": True, "raw_response": content, "parsed_result": parsed}

    except json.JSONDecodeError as exc:
        return {"success": False, "raw_response": content, "error": f"JSON parse error: {exc}"}
    except Exception as exc:
        return {"success": False, "raw_response": content, "error": f"LLM call failed: {exc}"}


def process_sample(
    sample: Any,
    sample_idx: int,
    client: Optional[Any],
    model: str,
    temperature: float,
    max_tokens: int,
    vehicle_length: float,
    vehicle_width: float,
    ego_cols: MatrixCols,
    neighbor_cols: MatrixCols,
    output_dir: Path,
    verbose: bool = True,
    save_prompt: bool = True,
    skip_llm: bool = False,
    enable_thinking: bool = False,
) -> Dict[str, Any]:
    """Generate a prompt for one sample and optionally call the LLM."""

    prompt = generate_cot_prompt(sample, vehicle_length, vehicle_width, ego_cols, neighbor_cols)
    if prompt is None:
        return {"sample_idx": sample_idx, "success": False, "error": "Prompt generation failed."}

    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = output_dir / f"prompt_sample_{sample_idx}.txt"
    response_file = output_dir / f"response_sample_{sample_idx}.txt"

    if save_prompt:
        prompt_file.write_text(prompt, encoding="utf-8")
        print(f"Saved prompt to: {prompt_file}")

    if verbose:
        print("=" * 80)
        print(f"Processing sample #{sample_idx}")
        print("=" * 80)
        print("\nPrompt preview (first 2000 characters):")
        print(prompt[:2000])
        if save_prompt:
            print(f"\nFull prompt length: {len(prompt)} characters. Full prompt: {prompt_file}")

    if skip_llm:
        return {"sample_idx": sample_idx, "prompt": prompt, "success": True, "skipped_llm": True}
    if client is None:
        raise RuntimeError("A client is required unless skip_llm=True.")

    llm_result = call_llm(prompt, client, model, temperature, max_tokens, enable_thinking=enable_thinking)

    if save_prompt:
        with response_file.open("w", encoding="utf-8") as f:
            if llm_result.get("success"):
                f.write("=== Parsed JSON result ===\n")
                f.write(json.dumps(llm_result["parsed_result"], indent=2, ensure_ascii=False))
            f.write("\n\n=== Raw LLM response ===\n")
            f.write(llm_result.get("raw_response", ""))
        print(f"Saved LLM response to: {response_file}")

    if verbose:
        print("\nLLM response:")
        if llm_result.get("success"):
            print(json.dumps(llm_result["parsed_result"], indent=2, ensure_ascii=False))
        else:
            print(f"Error: {llm_result.get('error', 'unknown error')}")
            print(llm_result.get("raw_response", "")[:500])

    return {"sample_idx": sample_idx, "prompt": prompt, **llm_result}


def normalize_speed_change(value: Any) -> str:
    """Normalize speed-change text to ACCEL, KEEP, or DECEL."""

    text = str(value or "KEEP").upper()
    if "DECEL" in text or "SLOW" in text or "BRAKE" in text:
        return "DECEL"
    if "ACCEL" in text or "SPEED UP" in text or "CATCH" in text:
        return "ACCEL"
    return "KEEP"


def extract_cvae_labels(llm_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract compact downstream labels from a parsed LLM response."""

    if not llm_result.get("success") or "parsed_result" not in llm_result:
        return None

    parsed = llm_result.get("parsed_result", {})
    decision = parsed.get("decision", {})
    prediction = parsed.get("prediction_guidance", {})
    reasoning = parsed.get("reasoning", {})
    phase = str(parsed.get("phase", "ANTICIPATION")).upper()

    if phase == "RELAXATION":
        strategy_mapping = {name: i for i, name in enumerate(RELAXATION_STRATEGIES)}
        default_strategy = "MAINTAIN_STABLE"
        gap_mapping = {"N/A": -1}
        target_gap = "N/A"
    else:
        strategy_mapping = {name: i for i, name in enumerate(MERGE_STRATEGIES)}
        default_strategy = "MAINTAIN_AND_OBSERVE"
        gap_mapping = {"GAP1": 0, "GAP2": 1, "EITHER": 2}
        target_gap = decision.get("target_gap", "EITHER")

    primary_strategy = str(decision.get("primary_strategy", default_strategy)).upper()
    result = {
        "phase": phase,
        "primary_strategy_id": strategy_mapping.get(primary_strategy, strategy_mapping[default_strategy]),
        "primary_strategy_name": primary_strategy,
        "strategy_scores": decision.get("strategy_scores", {}),
        "speed_change": normalize_speed_change(prediction.get("expected_speed_change")),
        "lateral_intent": decision.get("lateral_intent", 0.0),
        "longitudinal_intent": decision.get("longitudinal_intent", 0.0),
        "confidence": decision.get("confidence", 0.5),
        "expected_lateral_displacement": prediction.get("expected_lateral_displacement_3s", "0"),
        "key_interaction": prediction.get("key_interaction", "None"),
        "reasoning_summary": reasoning,
    }

    if phase == "RELAXATION":
        result.update(
            {
                "stability_level": decision.get("stability_level", 0.5),
                "target_speed": prediction.get("target_speed", "N/A"),
                "time_to_full_stability": prediction.get("time_to_full_stability", "3s"),
            }
        )
    else:
        result.update(
            {
                "target_gap_id": gap_mapping.get(str(target_gap).upper(), 2),
                "target_gap_name": target_gap,
                "aggressiveness": decision.get("aggressiveness", 0.5),
                "risk_tolerance": decision.get("risk_tolerance", 0.5),
                "gap_switch_probability": prediction.get("gap_switch_probability", 0.0),
                "lane_change_completion_probability": prediction.get("lane_change_completion_probability", 0.5),
            }
        )

    return result


def print_demo() -> None:
    """Print the three paper-defined reasoning phases when no data file exists."""

    print("Data file not found. Demo mode only.")
    print("Paper-aligned CoT reasoning phases:")
    print("1. ANTICIPATION: ego is preparing to change lanes; analyze TL, TF, and TFF.")
    print("2. CROSSING: ego is crossing the lane boundary; analyze whether it can complete safely.")
    print("3. RELAXATION: ego has entered the target lane; analyze post-merge stabilization with OL/OF.")


def env_float(name: str) -> Optional[float]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return float(value)


def env_int(name: str) -> Optional[int]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return int(value)


def optional_index(value: Optional[int]) -> Optional[int]:
    if value is None or value < 0:
        return None
    return value


def build_column_schemas(args: argparse.Namespace) -> tuple[Optional[MatrixCols], Optional[MatrixCols]]:
    required = [
        args.ego_x_col,
        args.ego_y_col,
        args.ego_lon_v_col,
        args.ego_lat_v_col,
        args.ego_acc_col,
        args.neighbor_x_col,
        args.neighbor_y_col,
        args.neighbor_lon_v_col,
        args.neighbor_lat_v_col,
        args.neighbor_acc_col,
    ]
    if any(value is None for value in required):
        return None, None

    ego_cols = MatrixCols(
        x=args.ego_x_col,
        y=args.ego_y_col,
        lon_v=args.ego_lon_v_col,
        lat_v=args.ego_lat_v_col,
        acc=args.ego_acc_col,
        yaw=optional_index(args.ego_yaw_col),
        lane_u=optional_index(args.ego_lane_u_col),
    )
    neighbor_cols = MatrixCols(
        x=args.neighbor_x_col,
        y=args.neighbor_y_col,
        lon_v=args.neighbor_lon_v_col,
        lat_v=args.neighbor_lat_v_col,
        acc=args.neighbor_acc_col,
        yaw=optional_index(args.neighbor_yaw_col),
    )
    return ego_cols, neighbor_cols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LLM CoT strategy labels from trajectory samples.")
    parser.add_argument("--data-path", type=Path, default=Path("data/train_dataset.mat"))
    parser.add_argument("--data-key", type=str, default="train_data")
    parser.add_argument("--sample-idx", type=int, default=10)
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Process every sample in the selected MAT split in zero-based row order.",
    )
    parser.add_argument("--random-sample", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/llm_cot"))
    parser.add_argument("--model", type=str, default="qwen-plus")
    parser.add_argument("--base-url", type=str, default=os.getenv("LLM_BASE_URL"))
    parser.add_argument("--api-key-env", type=str, default="LLM_API_KEY")
    parser.add_argument("--vehicle-length", type=float, default=env_float("COTTP_VEHICLE_LENGTH"))
    parser.add_argument("--vehicle-width", type=float, default=env_float("COTTP_VEHICLE_WIDTH"))
    parser.add_argument("--ego-x-col", type=int, default=env_int("COTTP_EGO_X_COL"))
    parser.add_argument("--ego-y-col", type=int, default=env_int("COTTP_EGO_Y_COL"))
    parser.add_argument("--ego-lon-v-col", type=int, default=env_int("COTTP_EGO_LON_V_COL"))
    parser.add_argument("--ego-lat-v-col", type=int, default=env_int("COTTP_EGO_LAT_V_COL"))
    parser.add_argument("--ego-acc-col", type=int, default=env_int("COTTP_EGO_ACC_COL"))
    parser.add_argument("--ego-yaw-col", type=int, default=env_int("COTTP_EGO_YAW_COL"))
    parser.add_argument("--ego-lane-u-col", type=int, default=env_int("COTTP_EGO_LANE_U_COL"))
    parser.add_argument("--neighbor-x-col", type=int, default=env_int("COTTP_NEIGHBOR_X_COL"))
    parser.add_argument("--neighbor-y-col", type=int, default=env_int("COTTP_NEIGHBOR_Y_COL"))
    parser.add_argument("--neighbor-lon-v-col", type=int, default=env_int("COTTP_NEIGHBOR_LON_V_COL"))
    parser.add_argument("--neighbor-lat-v-col", type=int, default=env_int("COTTP_NEIGHBOR_LAT_V_COL"))
    parser.add_argument("--neighbor-acc-col", type=int, default=env_int("COTTP_NEIGHBOR_ACC_COL"))
    parser.add_argument("--neighbor-yaw-col", type=int, default=env_int("COTTP_NEIGHBOR_YAW_COL"))
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--skip-llm", action="store_true", help="Only generate and save prompts; do not call the LLM.")
    parser.add_argument("--no-save-prompt", action="store_false", dest="save_prompt")
    parser.set_defaults(save_prompt=True)
    return parser.parse_args()


def require_clean_batch_output(output_dir: Path) -> None:
    """Prevent responses from an earlier dataset run entering a new batch."""

    if not output_dir.exists():
        return
    stale = []
    for pattern in ("prompt_sample_*.txt", "response_sample_*.txt", "batch_summary.json"):
        stale.extend(output_dir.glob(pattern))
    if stale:
        raise FileExistsError(
            f"Batch output directory contains prior sample artifacts: {output_dir}. "
            "Choose a new or clean directory for this split."
        )


def main(args: Optional[argparse.Namespace] = None) -> None:
    args = args or parse_args()

    print("=" * 80)
    print("CoT-TP LLM teacher reasoning")
    print("=" * 80)

    if not args.data_path.exists():
        print(f"Data file not found: {args.data_path}")
        print_demo()
        return

    try:
        samples = load_mat_data(args.data_path, args.data_key)
        print(f"Loaded {len(samples)} samples from {args.data_path} [{args.data_key}]")
    except Exception as exc:
        print(f"Failed to load data: {exc}")
        return
    if len(samples) == 0:
        raise ValueError(f"No samples found in {args.data_path} [{args.data_key}]")

    if args.all_samples and args.random_sample:
        raise ValueError("--all-samples and --random-sample cannot be used together")
    if args.all_samples:
        require_clean_batch_output(args.output_dir)

    if args.random_sample:
        sample_idx = random.randint(0, len(samples) - 1)
    else:
        sample_idx = args.sample_idx

    if not args.all_samples and (sample_idx < 0 or sample_idx >= len(samples)):
        print(f"Sample index out of range. Valid range: 0 to {len(samples) - 1}")
        return

    if args.vehicle_length is None or args.vehicle_width is None:
        print("Please provide --vehicle-length and --vehicle-width.")
        print("Alternatively set COTTP_VEHICLE_LENGTH and COTTP_VEHICLE_WIDTH.")
        return

    ego_cols, neighbor_cols = build_column_schemas(args)
    if ego_cols is None or neighbor_cols is None:
        print("Please provide the ego and surrounding-vehicle column mappings.")
        print("Use --ego-*-col and --neighbor-*-col arguments, or the matching COTTP_* environment variables.")
        return

    client = None
    if not args.skip_llm:
        try:
            client = make_client(args.api_key_env, args.base_url)
        except Exception as exc:
            print(f"Cannot create LLM client: {exc}")
            print("Use --skip-llm to generate prompts without making API calls.")
            return

    indices = range(len(samples)) if args.all_samples else [sample_idx]
    batch_records = []
    status_names = {0: "ANTICIPATION", 1: "CROSSING", 2: "RELAXATION"}
    for position, index in enumerate(indices, start=1):
        parsed = parse_sample(samples[index])
        print(
            f"Selected sample #{index}; lane_status={parsed.get('status', 0)} "
            f"({status_names.get(parsed.get('status', 0), 'UNKNOWN')})"
        )
        if args.all_samples:
            print(f"Batch progress: {position}/{len(samples)}")
        result = process_sample(
            samples[index],
            index,
            client=client,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            vehicle_length=args.vehicle_length,
            vehicle_width=args.vehicle_width,
            ego_cols=ego_cols,
            neighbor_cols=neighbor_cols,
            output_dir=args.output_dir,
            verbose=not args.all_samples,
            save_prompt=args.save_prompt,
            skip_llm=args.skip_llm,
            enable_thinking=args.enable_thinking,
        )
        batch_records.append(
            {
                "sample_idx": index,
                "success": bool(result.get("success", False)),
                "skipped_llm": bool(result.get("skipped_llm", False)),
            }
        )

        if not args.all_samples:
            labels = extract_cvae_labels(result)
            if labels is not None:
                print("\nDownstream labels:")
                print(json.dumps(labels, indent=2, ensure_ascii=False))

    if args.all_samples:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = args.output_dir / "batch_summary.json"
        summary_path.write_text(
            json.dumps(batch_records, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Saved batch summary to: {summary_path}")

    print("\nDone.")


def run_sample(idx: int) -> None:
    """Convenience helper for interactive use."""

    args = parse_args()
    args.sample_idx = idx
    main(args)


def batch_process(
    data_path: str,
    output_path: str,
    vehicle_length: float,
    vehicle_width: float,
    ego_cols: MatrixCols,
    neighbor_cols: MatrixCols,
    data_key: str = "train_data",
    num_samples: Optional[int] = None,
    skip_llm: bool = True,
) -> None:
    """Batch prompt generation helper. API calling is disabled by default."""

    samples = load_mat_data(Path(data_path), data_key)
    indices = random.sample(range(len(samples)), min(num_samples, len(samples))) if num_samples else range(len(samples))
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i, idx in enumerate(indices):
        print(f"Batch progress: {i + 1}/{len(indices)}")
        result = process_sample(
            samples[idx],
            idx,
            client=None,
            model="qwen-plus",
            temperature=0.3,
            max_tokens=2000,
            vehicle_length=vehicle_length,
            vehicle_width=vehicle_width,
            ego_cols=ego_cols,
            neighbor_cols=neighbor_cols,
            output_dir=out_dir,
            verbose=False,
            save_prompt=True,
            skip_llm=skip_llm,
        )
        records.append({"sample_idx": idx, "success": result.get("success", False), "prompt_file": f"prompt_sample_{idx}.txt"})

    summary_path = out_dir / "batch_summary.json"
    summary_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Saved batch summary to: {summary_path}")


if __name__ == "__main__":
    main()
