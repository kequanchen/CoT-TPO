from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


BASELINE_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from direct_llm.client import CompletionResult, DirectLLMAPIError  # noqa: E402
from direct_llm.generation import (  # noqa: E402
    PromptRecord,
    format_reminder,
    load_prompt_records,
    run_predictions,
)


class GenerationTests(unittest.TestCase):
    def test_loader_rejects_ground_truth_and_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prompts.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "sample_id": "one",
                        "system_prompt": "system",
                        "user_prompt": "observation",
                        "answer": "private future",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "ground-truth fields"):
                load_prompt_records(path)

            row = {"sample_id": "same", "system_prompt": "s", "user_prompt": "u"}
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
                load_prompt_records(path)

    def test_format_retry_is_fixed_gt_free_and_audited(self) -> None:
        record = PromptRecord("sample", "system", "observations only")
        seen_reminders = []
        outputs = iter(["not json", '{"future_trajectory":[[1,2]]}'])

        def completion(_record, reminder):
            seen_reminders.append(reminder)
            return CompletionResult(next(outputs), api_attempts=1)

        def parser(text):
            payload = json.loads(text)
            return payload

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "predictions.jsonl"
            counts = run_predictions(
                [record],
                output,
                completion,
                parser,
                expected_points=1,
                max_format_retries=2,
            )
            row = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(counts["ok"], 1)
        self.assertEqual(seen_reminders[0], None)
        self.assertEqual(seen_reminders[1], format_reminder(1))
        self.assertNotIn("[1,2]", seen_reminders[1])
        self.assertEqual([item["status"] for item in row["attempts"]], ["parse_error", "ok"])
        self.assertEqual(row["trajectory"], [[1, 2]])
        self.assertNotIn("future_trajectory", row)

    def test_api_error_is_safe_and_does_not_stop_later_samples(self) -> None:
        records = [PromptRecord("bad", "s", "u"), PromptRecord("good", "s", "u")]

        def completion(record, _reminder):
            if record.sample_id == "bad":
                raise DirectLLMAPIError("API request failed: HTTP 503", attempts=3)
            return CompletionResult('{"future_trajectory":[[0,0]]}', api_attempts=1)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "predictions.jsonl"
            counts = run_predictions(
                records,
                output,
                completion,
                json.loads,
                expected_points=1,
                max_format_retries=0,
            )
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(counts["api_errors"], 1)
        self.assertEqual(counts["ok"], 1)
        self.assertEqual([row["status"] for row in rows], ["api_error", "ok"])

    def test_resume_skips_every_existing_id_without_duplicates(self) -> None:
        records = [PromptRecord(str(index), "s", "u") for index in range(2)]
        calls = []

        def completion(record, _reminder):
            calls.append(record.sample_id)
            return CompletionResult('{"future_trajectory":[[0,0]]}', api_attempts=1)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "predictions.jsonl"
            run_predictions(
                records,
                output,
                completion,
                json.loads,
                expected_points=1,
                max_format_retries=0,
            )
            records.append(PromptRecord("2", "s", "u"))
            counts = run_predictions(
                records,
                output,
                completion,
                json.loads,
                expected_points=1,
                max_format_retries=0,
            )
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(counts["skipped"], 2)
        self.assertEqual(calls, ["0", "1", "2"])
        self.assertEqual([row["sample_id"] for row in rows], ["0", "1", "2"])

    def test_cli_help_requires_no_key_or_network(self) -> None:
        result = subprocess.run(
            [sys.executable, str(BASELINE_ROOT / "scripts" / "predict.py"), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage:", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
