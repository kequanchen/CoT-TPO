from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

BASELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINE_ROOT))

from lc_llm import (  # noqa: E402
    RoadConfig,
    adapt_sample,
    build_prompt,
    build_user_prompt,
    positions_to_local,
)


def synthetic_sample(*, lane_status: int = 0, time_since: float = -1.0) -> dict:
    steps = 10
    x = np.arange(steps, dtype=np.float32)
    y = np.zeros(steps, dtype=np.float32)
    x_hist = np.stack(
        [x, y, np.full(steps, 10.0), np.zeros(steps), np.zeros(steps), np.zeros(steps)],
        axis=1,
    )
    ego = np.zeros((steps, 19), dtype=np.float32)
    ego[:, 0] = 7
    ego[:, 1] = 4
    ego[:, 2] = np.arange(100, 100 + steps)
    ego[:, 3] = x
    ego[:, 4] = y
    ego[:, 5:7] = 2.0
    ego[:, 16] = 10.0

    neighbor = np.zeros((steps, 8), dtype=np.float32)
    neighbor[:, 0] = 11
    neighbor[:, 1] = 3
    neighbor[:, 2] = x + 12.0
    neighbor[:, 3] = 3.0
    neighbor[:, 4] = 8.0
    missing = np.full((steps, 8), np.nan, dtype=np.float32)
    return {
        "scenario_id": "synthetic_event",
        "traj_id": 9,
        "lane_status": lane_status,
        "time_since_crossing": time_since,
        "x_hist": x_hist,
        "y_future": np.full((50, 2), 9999.0, dtype=np.float32),
        "ctx": {
            "ego": ego,
            "phys_tl": neighbor,
            "phys_tf": missing,
            "phys_tff": missing,
            "phys_ol": missing,
            "phys_of": missing,
            "phys_off": missing,
        },
    }


class DataPromptTests(unittest.TestCase):
    def test_future_is_separate_and_prompt_is_observation_only(self) -> None:
        sample = adapt_sample(synthetic_sample())
        self.assertEqual(sample.future.shape, (50, 2))
        self.assertFalse(hasattr(sample.observation, "future"))
        prompt = build_prompt(sample.observation)
        self.assertNotIn("9999", prompt)
        self.assertNotIn("lane_status", prompt.lower())
        self.assertNotIn("time_since_crossing", prompt.lower())
        self.assertNotIn("anticipation", prompt.lower())
        self.assertNotIn("crossing", prompt.lower())
        self.assertNotIn("relaxation", prompt.lower())
        with self.assertRaises(TypeError):
            build_prompt(sample)  # type: ignore[arg-type]

    def test_phase_metadata_does_not_change_prompt(self) -> None:
        first = adapt_sample(synthetic_sample(lane_status=0, time_since=-2.0)).observation
        second = adapt_sample(synthetic_sample(lane_status=2, time_since=4.0)).observation
        self.assertEqual(build_prompt(first), build_prompt(second))

    def test_units_local_history_and_neutral_neighbor_names(self) -> None:
        observation = adapt_sample(synthetic_sample()).observation
        prompt = build_user_prompt(observation)
        self.assertIn("v_x = 36.00", prompt)  # 10 m/s converted to km/h
        self.assertIn("(-9.00, 0.00)", prompt)
        self.assertIn("(0.00, 0.00)", prompt)
        self.assertIn("surrounding vehicle A", prompt)
        self.assertNotIn("phys_tl", prompt)
        self.assertNotIn("target-lane leader", prompt)

    def test_lane_position_is_not_inferred_without_mapping(self) -> None:
        observation = adapt_sample(synthetic_sample()).observation
        unknown = build_user_prompt(observation, RoadConfig(num_lanes=None))
        self.assertIn("observed lane ID 4", unknown)
        self.assertIn("road position are unknown", unknown)
        mapped = build_user_prompt(
            observation,
            RoadConfig(num_lanes=5, lane_id_to_position=((4, "second-from-right"),)),
        )
        self.assertIn("5-lane highway", mapped)
        self.assertIn("second-from-right lane", mapped)

    def test_heading_aligned_coordinates(self) -> None:
        raw = synthetic_sample()
        raw["x_hist"] = raw["x_hist"].copy()
        raw["ctx"]["ego"] = raw["ctx"]["ego"].copy()
        raw["x_hist"][:, 0] = 0.0
        raw["x_hist"][:, 1] = np.arange(10, dtype=np.float32)
        raw["x_hist"][:, 5] = np.pi / 2
        raw["ctx"]["ego"][:, 3] = 0.0
        raw["ctx"]["ego"][:, 4] = np.arange(10, dtype=np.float32)
        observation = adapt_sample(raw).observation
        local = positions_to_local(np.asarray([[0.0, 10.0], [-1.0, 9.0]]), observation)
        np.testing.assert_allclose(local, [[1.0, 0.0], [0.0, 1.0]], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
