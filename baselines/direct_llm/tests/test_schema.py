from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


BASELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINE_ROOT))

from direct_llm import (  # noqa: E402
    DirectOutputParseError,
    format_direct_output,
    parse_direct_output,
)


class SchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trajectory = np.stack(
            [np.linspace(0.2, 10.0, 50), np.linspace(0.0, 2.0, 50)], axis=1
        ).astype(np.float32)

    def test_round_trip_exactly_one_trajectory(self) -> None:
        text = format_direct_output(self.trajectory, precision=3)
        parsed = parse_direct_output(text)
        self.assertEqual(parsed.trajectory.shape, (50, 2))
        np.testing.assert_allclose(parsed.trajectory, self.trajectory, atol=5.1e-4)

    def test_rejects_wrong_count_without_resampling(self) -> None:
        text = json.dumps({"future_trajectory": self.trajectory[:49].tolist()})
        with self.assertRaises(DirectOutputParseError):
            parse_direct_output(text)

    def test_rejects_extra_text_markdown_and_extra_keys(self) -> None:
        valid = format_direct_output(self.trajectory)
        for text in (
            "Here is the answer: " + valid,
            "```json\n" + valid + "\n```",
            valid + "\nExplanation: done",
            json.dumps(
                {"future_trajectory": self.trajectory.tolist(), "intention": 1}
            ),
        ):
            with self.subTest(text=text[:20]):
                with self.assertRaises(DirectOutputParseError):
                    parse_direct_output(text)

    def test_rejects_multiple_or_nested_candidate_trajectories(self) -> None:
        alternatives = [self.trajectory.tolist(), self.trajectory.tolist()]
        with self.assertRaises(DirectOutputParseError):
            parse_direct_output(json.dumps({"future_trajectory": alternatives}))

    def test_rejects_nonfinite_boolean_and_duplicate_key(self) -> None:
        payload = self.trajectory.tolist()
        payload[0][0] = True
        with self.assertRaises(DirectOutputParseError):
            parse_direct_output(json.dumps({"future_trajectory": payload}))
        nonfinite = format_direct_output(self.trajectory).replace("0.2", "NaN", 1)
        with self.assertRaises(DirectOutputParseError):
            parse_direct_output(nonfinite)
        duplicate = (
            '{"future_trajectory":'
            + json.dumps(self.trajectory.tolist())
            + ',"future_trajectory":'
            + json.dumps(self.trajectory.tolist())
            + "}"
        )
        with self.assertRaises(DirectOutputParseError):
            parse_direct_output(duplicate)


if __name__ == "__main__":
    unittest.main()
