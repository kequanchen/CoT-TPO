from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


BASELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINE_ROOT))

from direct_llm import (  # noqa: E402
    PromptConfig,
    adapt_observation,
    adapt_sample,
    build_prompt,
    build_user_prompt,
    positions_to_local,
)


def synthetic_sample() -> dict:
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

    near = np.zeros((steps, 8), dtype=np.float32)
    near[:, 0] = 11
    near[:, 2] = x + 3.0
    near[:, 3] = 1.0
    far = np.zeros((steps, 8), dtype=np.float32)
    far[:, 0] = 12
    far[:, 2] = x + 20.0
    far[:, 3] = -2.0
    missing = np.full((steps, 8), np.nan, dtype=np.float32)
    return {
        "scenario_id": "synthetic_event",
        "traj_id": 9,
        # Poison labels prove observation adaptation does not use them.
        "lane_status": 999,
        "time_since_crossing": 9876.5,
        "y_future": np.full((50, 2), 54321.0, dtype=np.float32),
        "x_hist": x_hist,
        "ctx": {
            "ego": ego,
            "phys_tl": far,
            "phys_tf": near,
            "phys_tff": missing,
            "phys_ol": missing,
            "phys_of": missing,
            "phys_off": missing,
        },
    }


class DataPromptTests(unittest.TestCase):
    def test_observation_type_physically_excludes_labels(self) -> None:
        observation = adapt_observation(synthetic_sample())
        self.assertFalse(hasattr(observation, "future"))
        self.assertFalse(hasattr(observation, "lane_status"))
        self.assertFalse(hasattr(observation, "time_since_crossing"))
        prompt = build_prompt(observation)
        self.assertNotIn("54321", prompt)
        self.assertNotIn("9876", prompt)
        self.assertNotIn("lane_status", prompt)
        self.assertNotIn("time_since_crossing", prompt)

    def test_future_is_opt_in_and_never_accepted_by_prompt(self) -> None:
        without = adapt_sample(synthetic_sample(), include_future=False)
        with_future = adapt_sample(synthetic_sample(), include_future=True)
        self.assertIsNone(without.future)
        self.assertEqual(with_future.future.shape, (50, 2))
        with self.assertRaises(TypeError):
            build_prompt(with_future)  # type: ignore[arg-type]

    def test_target_history_is_local_and_has_ten_points(self) -> None:
        prompt = build_user_prompt(adapt_observation(synthetic_sample()))
        self.assertIn("[-9.0,0.0]", prompt)
        self.assertIn("[0.0,0.0]", prompt)
        target_line = prompt.split("Target observed trajectory:\n", 1)[1].splitlines()[0]
        import json

        self.assertEqual(len(json.loads(target_line)), 10)

    def test_neighbors_are_geometry_sorted_and_source_roles_are_hidden(self) -> None:
        raw = synthetic_sample()
        first = build_user_prompt(adapt_observation(raw))
        swapped = synthetic_sample()
        swapped["ctx"]["phys_tl"], swapped["ctx"]["phys_tf"] = (
            swapped["ctx"]["phys_tf"],
            swapped["ctx"]["phys_tl"],
        )
        second = build_user_prompt(adapt_observation(swapped))
        self.assertEqual(first, second)
        self.assertIn("Vehicle A", first)
        self.assertIn("Vehicle B", first)
        self.assertNotIn("phys_tl", first)
        self.assertNotIn("phys_tf", first)
        self.assertNotIn("leader", first.lower())
        self.assertLess(first.index("Vehicle A time"), first.index("Vehicle B time"))

    def test_partial_neighbor_history_retains_time_offsets(self) -> None:
        raw = synthetic_sample()
        raw["ctx"]["phys_tf"] = raw["ctx"]["phys_tf"].copy()
        raw["ctx"]["phys_tf"][:2] = np.nan
        prompt = build_user_prompt(adapt_observation(raw))
        self.assertIn(
            "Vehicle A time offsets (s): [-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0]",
            prompt,
        )

    def test_target_only_context_omits_surrounding_trajectories(self) -> None:
        observation = adapt_observation(synthetic_sample())
        prompt = build_user_prompt(
            observation,
            PromptConfig(context_mode="target_only", trajectory_precision=2, max_neighbors=6),
        )
        self.assertNotIn("Vehicle A", prompt)
        self.assertIn("Use only the target trajectory", prompt)

    def test_heading_aligned_forward_left_coordinates(self) -> None:
        raw = synthetic_sample()
        raw["x_hist"] = raw["x_hist"].copy()
        raw["ctx"]["ego"] = raw["ctx"]["ego"].copy()
        raw["x_hist"][:, 0] = 0.0
        raw["x_hist"][:, 1] = np.arange(10, dtype=np.float32)
        raw["x_hist"][:, 5] = np.pi / 2
        raw["ctx"]["ego"][:, 3] = 0.0
        raw["ctx"]["ego"][:, 4] = np.arange(10, dtype=np.float32)
        observation = adapt_observation(raw)
        local = positions_to_local(np.asarray([[0.0, 10.0], [-1.0, 9.0]]), observation)
        np.testing.assert_allclose(local, [[1.0, 0.0], [0.0, 1.0]], atol=1e-5)

    def test_sample_id_is_deterministic(self) -> None:
        observation = adapt_observation(synthetic_sample())
        self.assertEqual(observation.sample_id, "synthetic_event:9:109")


if __name__ == "__main__":
    unittest.main()
