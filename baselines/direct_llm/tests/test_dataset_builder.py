from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


BASELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINE_ROOT))

from direct_llm import (  # noqa: E402
    adapt_observation,
    build_inference_record,
    build_records,
    write_jsonl,
)
from test_data_prompt import synthetic_sample  # noqa: E402


class DatasetBuilderTests(unittest.TestCase):
    def test_record_has_prompt_fields_only(self) -> None:
        observation = adapt_observation(synthetic_sample())
        record = build_inference_record(observation)
        self.assertEqual(
            set(record),
            {
                "sample_id",
                "event_id",
                "source_split",
                "system_prompt",
                "user_prompt",
                "prompt_text",
            },
        )
        serialized = json.dumps(record).lower()
        for forbidden in (
            '"answer"',
            '"y_future"',
            '"future_local"',
            '"lane_status"',
            '"time_since_crossing"',
            "54321",
            "9876",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_only_test_split_is_allowed(self) -> None:
        observation = adapt_observation(synthetic_sample())
        with self.assertRaises(ValueError):
            build_inference_record(observation, source_split="train")

    def test_jsonl_writer_and_duplicate_guard(self) -> None:
        observation = adapt_observation(synthetic_sample())
        record = build_records([observation])[0]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prompts.jsonl"
            write_jsonl([record], output)
            loaded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(loaded["sample_id"], "synthetic_event:9:109")
            self.assertNotIn("answer", loaded)
            with self.assertRaises(FileExistsError):
                write_jsonl([record], output)
            with self.assertRaises(ValueError):
                write_jsonl([record, record], Path(directory) / "duplicates.jsonl")

    def test_writer_rejects_injected_ground_truth_field(self) -> None:
        record = build_inference_record(adapt_observation(synthetic_sample()))
        record["y_future"] = [[1, 2]]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_jsonl([record], Path(directory) / "unsafe.jsonl")


if __name__ == "__main__":
    unittest.main()
