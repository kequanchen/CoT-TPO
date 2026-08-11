from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

BASELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINE_ROOT))

from lc_llm import (  # noqa: E402
    LCOutputParseError,
    format_lc_llm_output,
    parse_lc_llm_output,
)


class SchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trajectory = np.stack(
            [np.linspace(0.2, 10.0, 50), np.linspace(0.0, 3.0, 50)], axis=1
        ).astype(np.float32)

    def test_format_parse_round_trip(self) -> None:
        answer = format_lc_llm_output(
            ["the vehicle ahead is slower"],
            ["change to the left lane"],
            1,
            self.trajectory,
            precision=3,
        )
        parsed = parse_lc_llm_output(answer)
        self.assertEqual(parsed.intention, 1)
        self.assertEqual(parsed.trajectory.shape, (50, 2))
        self.assertEqual(parsed.notable_features, ("the vehicle ahead is slower",))
        np.testing.assert_allclose(parsed.trajectory, self.trajectory, atol=5.1e-4)

    def test_rejects_wrong_point_count(self) -> None:
        payload = json.dumps(self.trajectory[:49].tolist())
        text = (
            "Thought:\n- Notable feature: steady\n- Potential behavior: keep lane\n"
            'Final answer:\n- Intention: "0: Keep lane"\n'
            f'- Trajectory: "{payload}"'
        )
        with self.assertRaises(LCOutputParseError):
            parse_lc_llm_output(text)

    def test_rejects_mismatched_intention_label(self) -> None:
        payload = json.dumps(self.trajectory.tolist())
        text = (
            "Thought:\n- Notable feature: steady\n- Potential behavior: keep lane\n"
            'Final answer:\n- Intention: "1: Right lane change"\n'
            f'- Trajectory: "{payload}"'
        )
        with self.assertRaises(LCOutputParseError):
            parse_lc_llm_output(text)

    def test_rejects_non_finite_and_unrecognized_lines(self) -> None:
        payload = json.dumps(self.trajectory.tolist()).replace("0.0", "NaN", 1)
        bad = (
            "Thought:\n- Notable feature: steady\n- Potential behavior: keep lane\n"
            'Final answer:\n- Intention: "0: Keep lane"\n'
            f'- Trajectory: "{payload}"'
        )
        with self.assertRaises(LCOutputParseError):
            parse_lc_llm_output(bad)
        answer = format_lc_llm_output(["steady"], ["keep lane"], 0, self.trajectory)
        with self.assertRaises(LCOutputParseError):
            parse_lc_llm_output(answer + "\nExplanation: extra")


if __name__ == "__main__":
    unittest.main()
