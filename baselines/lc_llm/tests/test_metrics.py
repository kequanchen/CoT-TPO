import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


BASELINE_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from lc_llm.metrics import (  # noqa: E402
    CoverageError,
    ReferenceRecord,
    evaluate_predictions,
    intention_classification_metrics,
    load_prediction_jsonl,
    top1_ade_fde,
)


class MetricsTests(unittest.TestCase):
    def test_top1_ade_fde_at_multiple_horizons(self):
        truth = np.zeros((2, 50, 2), dtype=np.float64)
        prediction = np.zeros_like(truth)
        prediction[0, :, 0] = 1.0
        prediction[1, :, 1] = 3.0

        metrics = top1_ade_fde(
            prediction,
            truth,
            sample_rate_hz=10.0,
            horizons_seconds=[1, 5],
        )

        self.assertAlmostEqual(metrics["1s"]["ADE"], 2.0)
        self.assertAlmostEqual(metrics["1s"]["FDE"], 2.0)
        self.assertAlmostEqual(metrics["5s"]["ADE"], 2.0)
        self.assertEqual(metrics["5s"]["steps"], 50)

    def test_intention_accuracy_and_macro_f1(self):
        metrics = intention_classification_metrics(
            ["keep_lane", "left_lane_change", "left_lane_change"],
            ["keep", "left", "keep"],
        )

        self.assertAlmostEqual(metrics["accuracy"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["per_class"]["keep_lane"]["f1"], 2.0 / 3.0)
        self.assertAlmostEqual(
            metrics["per_class"]["left_lane_change"]["f1"], 2.0 / 3.0
        )
        self.assertAlmostEqual(metrics["macro_f1"], 2.0 / 3.0)
        self.assertEqual(metrics["labels"], ["keep_lane", "left_lane_change"])

    def test_jsonl_loader_retains_parse_failure_for_coverage(self):
        trajectory = np.zeros((50, 2)).tolist()
        records = [
            {
                "sample_id": "a",
                "status": "ok",
                "intention": "left lane change",
                "trajectory": trajectory,
            },
            {
                "sample_id": "b",
                "status": "parse_error",
                "raw_output": "not parseable",
                "error": "missing trajectory",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(
                "\n".join(json.dumps(record) for record in records), encoding="utf-8"
            )
            loaded = load_prediction_jsonl(path, expected_steps=50)

        self.assertEqual(loaded.attempted_ids, frozenset({"a", "b"}))
        self.assertEqual(set(loaded.valid), {"a"})
        self.assertIn("b", loaded.invalid)

    def test_strict_coverage_rejects_invalid_missing_and_extra(self):
        future = np.zeros((50, 2), dtype=np.float64)
        references = [
            ReferenceRecord("a", "left", future),
            ReferenceRecord("b", "keep", future),
        ]
        records = [
            {
                "sample_id": "a",
                "status": "ok",
                "intention": "left",
                "trajectory": future.tolist(),
            },
            {"sample_id": "b", "status": "parse_error", "error": "bad output"},
            {
                "sample_id": "extra",
                "status": "ok",
                "intention": "keep",
                "trajectory": future.tolist(),
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(
                "\n".join(json.dumps(record) for record in records), encoding="utf-8"
            )
            loaded = load_prediction_jsonl(path, expected_steps=50)

        with self.assertRaises(CoverageError) as raised:
            evaluate_predictions(
                references,
                loaded,
                expected_steps=50,
                sample_rate_hz=10.0,
                horizons_seconds=[1, 2, 3, 4, 5],
                require_full_coverage=True,
            )
        self.assertEqual(raised.exception.report["invalid_sample_ids"], ["b"])
        self.assertEqual(raised.exception.report["extra_sample_ids"], ["extra"])
        self.assertFalse(raised.exception.report["exact"])

    def test_sample_id_join_is_order_independent(self):
        truth_a = np.zeros((50, 2), dtype=np.float64)
        truth_b = np.ones((50, 2), dtype=np.float64)
        report = evaluate_predictions(
            [
                ReferenceRecord("a", "keep", truth_a),
                ReferenceRecord("b", "left", truth_b),
            ],
            [
                {"sample_id": "b", "intention": "left", "trajectory": truth_b},
                {"sample_id": "a", "intention": "keep", "trajectory": truth_a},
            ],
            expected_steps=50,
            sample_rate_hz=10.0,
            horizons_seconds=[1, 5],
        )

        self.assertEqual(report["coverage"]["valid"], 2)
        self.assertTrue(report["coverage"]["exact"])
        self.assertEqual(report["intention"]["accuracy"], 1.0)
        self.assertEqual(report["trajectory"]["5s"]["ADE"], 0.0)

    def test_rejects_duplicate_prediction_id(self):
        record = {
            "sample_id": "duplicate",
            "status": "ok",
            "intention": "keep",
            "trajectory": np.zeros((50, 2)).tolist(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(
                json.dumps(record) + "\n" + json.dumps(record), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
                load_prediction_jsonl(path, expected_steps=50)


if __name__ == "__main__":
    unittest.main()
