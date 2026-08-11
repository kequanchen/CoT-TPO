import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


BASELINE_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from direct_llm.metrics import (  # noqa: E402
    CoverageError,
    ReferenceRecord,
    evaluate_predictions,
    load_prediction_jsonl,
    top1_ade_fde,
)


class MetricsTests(unittest.TestCase):
    def test_top1_ade_fde_at_one_and_five_seconds(self):
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

    def test_jsonl_loader_retains_api_and_parse_failures(self):
        records = [
            {
                "sample_id": "ok",
                "status": "ok",
                "trajectory": np.zeros((50, 2)).tolist(),
                "raw_output": "[]",
            },
            {
                "sample_id": "parse",
                "status": "parse_error",
                "error": "wrong number of points",
                "raw_output": "bad",
            },
            {
                "sample_id": "api",
                "status": "api_error",
                "error": "request failed after retries",
                "raw_output": "",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(
                "\n".join(json.dumps(record) for record in records), encoding="utf-8"
            )
            loaded = load_prediction_jsonl(path, expected_steps=50)

        self.assertEqual(loaded.attempted_ids, frozenset({"ok", "parse", "api"}))
        self.assertEqual(set(loaded.valid), {"ok"})
        self.assertEqual(set(loaded.invalid), {"parse", "api"})

    def test_strict_coverage_rejects_failed_missing_and_extra_ids(self):
        future = np.zeros((50, 2), dtype=np.float64)
        references = [ReferenceRecord("a", future), ReferenceRecord("b", future)]
        records = [
            {"sample_id": "a", "status": "ok", "trajectory": future.tolist()},
            {"sample_id": "b", "status": "parse_error", "error": "bad"},
            {"sample_id": "extra", "status": "ok", "trajectory": future.tolist()},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(
                "\n".join(json.dumps(record) for record in records), encoding="utf-8"
            )
            predictions = load_prediction_jsonl(path, expected_steps=50)

        with self.assertRaises(CoverageError) as raised:
            evaluate_predictions(
                references,
                predictions,
                expected_steps=50,
                sample_rate_hz=10.0,
                horizons_seconds=[1, 2, 3, 4, 5],
            )
        self.assertEqual(raised.exception.report["invalid_sample_ids"], ["b"])
        self.assertEqual(raised.exception.report["extra_sample_ids"], ["extra"])
        self.assertFalse(raised.exception.report["exact"])

    def test_join_is_by_sample_id_not_file_order(self):
        truth_a = np.zeros((50, 2), dtype=np.float64)
        truth_b = np.ones((50, 2), dtype=np.float64)
        report = evaluate_predictions(
            [ReferenceRecord("a", truth_a), ReferenceRecord("b", truth_b)],
            [
                {"sample_id": "b", "trajectory": truth_b},
                {"sample_id": "a", "trajectory": truth_a},
            ],
            expected_steps=50,
            sample_rate_hz=10.0,
            horizons_seconds=[1, 5],
        )

        self.assertTrue(report["coverage"]["exact"])
        self.assertEqual(report["trajectory"]["1s"]["ADE"], 0.0)
        self.assertEqual(report["trajectory"]["5s"]["FDE"], 0.0)

    def test_non_strict_mode_reports_partial_denominator(self):
        future = np.zeros((50, 2), dtype=np.float64)
        report = evaluate_predictions(
            [ReferenceRecord("a", future), ReferenceRecord("b", future)],
            [{"sample_id": "a", "trajectory": future}],
            expected_steps=50,
            sample_rate_hz=10.0,
            horizons_seconds=[5],
            require_full_coverage=False,
        )

        self.assertEqual(report["coverage"]["valid"], 1)
        self.assertEqual(report["coverage"]["valid_fraction"], 0.5)
        self.assertEqual(report["coverage"]["missing_sample_ids"], ["b"])

    def test_rejects_duplicate_prediction_id(self):
        record = {
            "sample_id": "same",
            "status": "ok",
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
