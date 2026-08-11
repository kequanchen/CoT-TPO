from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

BASELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINE_ROOT))

from llc_pc import ContextParseError, encode_context, parse_llm_response  # noqa: E402


def valid_response() -> dict:
    return {
        "Situation Understanding": "The target is approaching a usable adjacent-lane gap.",
        "Reasoning": "The observed relative positions support a cautious lateral transition.",
        "Actions": ["STRAIGHT-LEFT", "STRAIGHT"],
        "Affordance": ["LEFT-ALLOW", "SLOW_ALLOW"],
        "Scenario_name": "ON-STRAIGHT-ROAD",
    }


class SchemaTests(unittest.TestCase):
    def test_strict_parse_and_17d_encoding(self) -> None:
        fenced = "```json\n" + json.dumps(valid_response()) + "\n```"
        parsed = parse_llm_response(fenced)
        self.assertEqual(parsed.actions, ("STRAIGHT_LEFT", "STRAIGHT"))
        vector = encode_context(parsed)
        self.assertEqual(vector.shape, (17,))
        self.assertEqual(vector.dtype, np.float32)
        self.assertEqual(vector[2], 2.0)
        self.assertEqual(vector[1], 1.0)
        np.testing.assert_array_equal(vector[8:12], [1.0, 0.0, 1.0, 0.0])
        np.testing.assert_array_equal(vector[12:17], [0.0, 1.0, 0.0, 0.0, 0.0])

    def test_rejects_extra_fields(self) -> None:
        response = valid_response()
        response["future_coordinates"] = [[1.0, 2.0]]
        with self.assertRaises(ContextParseError):
            parse_llm_response(response)

    def test_rejects_unknown_and_duplicate_labels(self) -> None:
        response = valid_response()
        response["Actions"] = ["MERGE_NOW"]
        with self.assertRaises(ContextParseError):
            parse_llm_response(response)
        response = valid_response()
        response["Affordance"] = ["LEFT_ALLOW", "LEFT-ALLOW"]
        with self.assertRaises(ContextParseError):
            parse_llm_response(response)

    def test_requires_json_arrays(self) -> None:
        response = valid_response()
        response["Actions"] = "STRAIGHT_LEFT, STRAIGHT"
        with self.assertRaises(ContextParseError):
            parse_llm_response(response)


if __name__ == "__main__":
    unittest.main()
