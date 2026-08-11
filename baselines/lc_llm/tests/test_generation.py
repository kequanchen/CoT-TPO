from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path


BASELINE_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from lc_llm.generation import (  # noqa: E402
    GenerationSettings,
    generate_text_batch,
    prompt_for_inference,
    run_batched_prediction,
    tokenize_generation_prompts,
)
from lc_llm.modeling import validate_adapter_directory  # noqa: E402
from lc_llm.sft import SFTRecord  # noqa: E402


@dataclass(frozen=True)
class Parsed:
    intention: int
    trajectory: tuple[tuple[float, float], ...]


class GenerationTests(unittest.TestCase):
    def test_generation_budget_must_fit_model_context(self) -> None:
        settings = GenerationSettings(max_input_tokens=24, max_new_tokens=8)
        settings.validate_context_window(32)
        with self.assertRaisesRegex(ValueError, "exceeds model.max_seq_length"):
            settings.validate_context_window(31)

    def test_inference_prompt_cannot_leak_supervised_answer(self) -> None:
        secret = "FUTURE_SECRET_COORDINATES"
        record = SFTRecord("id", "system", "observed history", secret)
        prompt = prompt_for_inference(record)
        self.assertNotIn(secret, prompt)
        self.assertIn("observed history", prompt)

    def test_long_generation_prompt_retains_prefix_and_inst_suffix(self) -> None:
        class Tokenizer:
            def __init__(self):
                self.vocab = {}

            def encode(self, text, *, add_special_tokens=False):
                self.assert_special = add_special_tokens
                ids = []
                for word in text.split():
                    self.vocab.setdefault(word, len(self.vocab) + 1)
                    ids.append(self.vocab[word])
                return ids

        tokenizer = Tokenizer()
        record = SFTRecord(
            "long",
            "system contract remains visible",
            " ".join(f"observation-{index}" for index in range(100)),
            "unused",
        )
        token_ids = tokenize_generation_prompts(
            tokenizer,
            [record],
            max_input_tokens=16,
        )[0]
        self.assertEqual(len(token_ids), 16)
        self.assertEqual(token_ids[0], tokenizer.vocab["<s>[INST]"])
        self.assertEqual(token_ids[-1], tokenizer.vocab["[/INST]"])

    def test_generate_text_batch_decodes_only_continuations(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")

        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 2

            def encode(self, text, *, add_special_tokens=False):
                return list(range(10, 10 + len(text.split())))

            def batch_decode(self, rows, **_kwargs):
                return [" ".join(str(int(item)) for item in row) for row in rows]

        class Model:
            device = torch.device("cpu")

            def generate(self, input_ids, attention_mask, **_kwargs):
                self.seen_attention = attention_mask
                continuation = torch.tensor([[91, 2], [92, 2]], dtype=torch.long)
                return torch.cat((input_ids, continuation), dim=1)

        records = [
            SFTRecord("a", "system", "short", "unused"),
            SFTRecord("b", "system", "a somewhat longer prompt", "unused"),
        ]
        texts = generate_text_batch(
            Model(),
            Tokenizer(),
            records,
            GenerationSettings(batch_size=2, max_input_tokens=64, max_new_tokens=8),
        )
        self.assertEqual(texts, ["91 2", "92 2"])

    def test_predictions_are_batched_and_resumable(self) -> None:
        records = [SFTRecord(str(i), "s", f"u{i}", "unused") for i in range(3)]
        calls: list[list[str]] = []

        def predictor(batch):
            calls.append([record.sample_id for record in batch])
            return [f"generated-{record.sample_id}" for record in batch]

        def parser(text):
            return Parsed(1, ((1.0, 2.0),))

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "predictions.jsonl"
            first = run_batched_prediction(
                records,
                output,
                predictor,
                batch_size=2,
                parser=parser,
            )
            self.assertEqual(first["written"], 3)
            self.assertEqual(calls, [["0", "1"], ["2"]])
            records.append(SFTRecord("3", "s", "u3", "unused"))
            second = run_batched_prediction(
                records,
                output,
                predictor,
                batch_size=2,
                parser=parser,
            )
            self.assertEqual(second["skipped"], 3)
            self.assertEqual(second["written"], 1)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["sample_id"] for row in rows], ["0", "1", "2", "3"])
            self.assertTrue(all(row["status"] == "ok" for row in rows))
            self.assertEqual(rows[0]["trajectory"], [[1.0, 2.0]])

    def test_parse_failure_is_auditable(self) -> None:
        record = SFTRecord("bad", "s", "u", "unused")

        def bad_parser(_text):
            raise ValueError("wrong number of points")

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "predictions.jsonl"
            summary = run_batched_prediction(
                [record],
                output,
                lambda batch: ["malformed"],
                batch_size=1,
                parser=bad_parser,
            )
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(summary["errors"], 1)
            self.assertEqual(row["status"], "parse_error")
            self.assertIn("wrong number of points", row["error"])
            self.assertEqual(row["raw_output"], "malformed")

    def test_adapter_validation_requires_local_safetensors_and_matching_base(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            adapter = Path(temporary)
            (adapter / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "peft_type": "LORA",
                        "task_type": "CAUSAL_LM",
                        "base_model_name_or_path": "meta-llama/Llama-2-13b-chat-hf",
                    }
                ),
                encoding="utf-8",
            )
            (adapter / "adapter_model.safetensors").write_bytes(b"safe-placeholder")
            # HF Trainer writes this metadata file at the adapter root.  It is
            # not a model weight candidate and must not break safe inference.
            (adapter / "training_args.bin").write_bytes(b"trainer-metadata")
            metadata = validate_adapter_directory(
                adapter,
                expected_base_model="meta-llama/Llama-2-13b-chat-hf",
            )
            self.assertEqual(metadata["peft_type"], "LORA")
            with self.assertRaisesRegex(ValueError, "does not match"):
                validate_adapter_directory(adapter, expected_base_model="another/model")
            (adapter / "adapter_model.bin").write_bytes(b"pickle-compatible")
            with self.assertRaisesRegex(ValueError, "pickle-compatible"):
                validate_adapter_directory(adapter)

    def test_cli_help_has_no_transformers_dependency(self) -> None:
        for script in ("train_lora.py", "predict.py"):
            result = subprocess.run(
                [sys.executable, str(BASELINE_ROOT / "scripts" / script), "--help"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
