from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

BASELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINE_ROOT))

from lc_llm import (  # noqa: E402
    LabelConfig,
    adapt_sample,
    build_inference_record,
    build_supervised_record,
    infer_training_intention,
    parse_lc_llm_output,
    write_jsonl,
)
from test_data_prompt import synthetic_sample  # noqa: E402


def labelled_sample(*, status: int, lateral_endpoint: float):
    raw = synthetic_sample(lane_status=status)
    current_x = float(raw["x_hist"][-1, 0])
    current_y = float(raw["x_hist"][-1, 1])
    raw["y_future"] = np.stack(
        [
            current_x + np.linspace(0.2, 10.0, 50, dtype=np.float32),
            current_y + np.linspace(0.0, lateral_endpoint, 50, dtype=np.float32),
        ],
        axis=1,
    )
    return adapt_sample(raw)


class DatasetBuilderTests(unittest.TestCase):
    def test_phase_left_event_labels(self) -> None:
        self.assertEqual(infer_training_intention(labelled_sample(status=0, lateral_endpoint=0.0)), 1)
        self.assertEqual(infer_training_intention(labelled_sample(status=1, lateral_endpoint=3.0)), 1)
        self.assertEqual(infer_training_intention(labelled_sample(status=2, lateral_endpoint=0.0)), 0)

    def test_optional_displacement_labels(self) -> None:
        config = LabelConfig("future_lateral_displacement", 1.5)
        self.assertEqual(
            infer_training_intention(labelled_sample(status=2, lateral_endpoint=2.0), config), 1
        )
        self.assertEqual(
            infer_training_intention(labelled_sample(status=0, lateral_endpoint=-2.0), config), 2
        )
        self.assertEqual(
            infer_training_intention(labelled_sample(status=0, lateral_endpoint=0.5), config), 0
        )

    def test_supervised_record_and_answer(self) -> None:
        sample = labelled_sample(status=0, lateral_endpoint=3.0)
        record = build_supervised_record(sample)
        required = {
            "sample_id",
            "event_id",
            "source_split",
            "system_prompt",
            "user_prompt",
            "prompt_text",
            "answer",
            "intention",
            "future_local",
        }
        self.assertEqual(set(record), required)
        self.assertEqual(record["source_split"], "train")
        self.assertNotIn(record["answer"], record["prompt_text"])
        parsed = parse_lc_llm_output(record["answer"])
        self.assertEqual(parsed.intention, 1)
        self.assertEqual(parsed.trajectory.shape, (50, 2))
        with self.assertRaises(ValueError):
            build_supervised_record(sample, source_split="test")

    def test_inference_record_has_no_future_or_answer(self) -> None:
        sample = labelled_sample(status=1, lateral_endpoint=3.0)
        record = build_inference_record(sample.observation)
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
        serialized = json.dumps(record)
        self.assertNotIn("future_local", record)
        self.assertNotIn("answer", record)
        self.assertNotIn("9999", serialized)
        self.assertNotIn("lane_status", serialized.lower())

    def test_jsonl_writer_and_duplicate_guard(self) -> None:
        record = build_inference_record(labelled_sample(status=2, lateral_endpoint=0.0).observation)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            write_jsonl([record], path)
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["sample_id"], record["sample_id"])
            with self.assertRaises(FileExistsError):
                write_jsonl([record], path)
            with self.assertRaises(ValueError):
                write_jsonl([record, record], Path(directory) / "duplicates.jsonl")


if __name__ == "__main__":
    unittest.main()
