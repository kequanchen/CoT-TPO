from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

BASELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINE_ROOT))

from llc_pc import (  # noqa: E402
    CONTEXT_DIM,
    SampleValidationError,
    adapt_sample,
    build_prompt,
    build_tc_map_scene,
    render_tc_map_png,
)


def synthetic_sample() -> dict:
    steps = 10
    x = np.linspace(0.0, 9.0, steps, dtype=np.float32)
    y = np.linspace(0.0, 0.45, steps, dtype=np.float32)
    x_hist = np.stack(
        [x, y, np.full(steps, 10.0), np.full(steps, 0.5), np.zeros(steps), np.zeros(steps)],
        axis=1,
    )
    ego = np.zeros((steps, 19), dtype=np.float32)
    ego[:, 0] = 7
    ego[:, 1] = 2
    ego[:, 2] = np.arange(steps)
    ego[:, 3] = x
    ego[:, 4] = y
    ego[:, 5] = 1.0
    ego[:, 6] = 3.0
    ego[:, 15] = 0.5
    ego[:, 16] = 10.0

    leader = np.zeros((steps, 8), dtype=np.float32)
    leader[:, 0] = 11
    leader[:, 1] = 1
    leader[:, 2] = x + 12.0
    leader[:, 3] = y - 4.0
    leader[:, 4] = 9.0
    missing = np.full((steps, 8), np.nan, dtype=np.float32)
    return {
        "scenario_id": "synthetic_scene",
        "traj_id": 3,
        "lane_status": 1,
        "time_since_crossing": 0.0,
        "x_hist": x_hist,
        "y_future": np.full((50, 2), 9999.0, dtype=np.float32),
        "ctx": {
            "ego": ego,
            "phys_tl": leader,
            "phys_tf": np.zeros((steps, 8), dtype=np.float32),
            "phys_tff": missing,
            "phys_ol": missing,
            "phys_of": missing,
            "phys_off": missing,
        },
    }


class DataAndPromptTests(unittest.TestCase):
    def test_adaptation_separates_future_and_builds_masks(self) -> None:
        sample = adapt_sample(synthetic_sample())
        self.assertEqual(sample.observation.x_hist.shape, (10, 6))
        self.assertEqual(sample.future.shape, (50, 2))
        self.assertTrue(sample.observation.neighbor_masks["phys_tl"].all())
        self.assertFalse(sample.observation.neighbor_masks["phys_tf"].any())
        self.assertTrue(np.all(sample.observation.neighbors["phys_tf"] == 0))

    def test_scene_lane_geometry_and_png(self) -> None:
        observation = adapt_sample(synthetic_sample()).observation
        scene = build_tc_map_scene(observation)
        np.testing.assert_allclose(scene.lane_boundaries_m, [-5.0, -1.0, 3.0, 7.0])
        self.assertAlmostEqual(scene.lane_width_m, 4.0)
        self.assertEqual(scene.traces["phys_tl"].highlight_rank, 0)
        png = render_tc_map_png(observation)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertGreater(len(png), 1000)

    def test_prompt_cannot_receive_future_sample(self) -> None:
        sample = adapt_sample(synthetic_sample())
        prompt = build_prompt(sample.observation)
        self.assertEqual(CONTEXT_DIM, 17)
        self.assertNotIn("9999", prompt)
        self.assertIn("ON_STRAIGHT_ROAD", prompt)
        self.assertIn("target vehicle is shown in red", prompt)
        with self.assertRaises(TypeError):
            build_prompt(sample)  # type: ignore[arg-type]

    def test_rejects_misaligned_histories(self) -> None:
        raw = synthetic_sample()
        raw["ctx"]["ego"] = raw["ctx"]["ego"].copy()
        raw["ctx"]["ego"][:, 3] += 1.0
        with self.assertRaises(SampleValidationError):
            adapt_sample(raw)


if __name__ == "__main__":
    unittest.main()
