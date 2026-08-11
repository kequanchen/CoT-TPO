from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


BASELINE_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from lc_llm.sft import (  # noqa: E402
    IGNORE_INDEX,
    AnswerOnlyDataCollator,
    SFTRecord,
    TokenizedSFTDataset,
    encode_sft_record,
    format_llama2_chat_prompt,
    load_jsonl_records,
)


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}
        self.encode_calls = 0

    def encode(self, text: str, *, add_special_tokens: bool = False):
        self.encode_calls += 1
        if add_special_tokens:
            raise AssertionError("SFT helpers must control special tokens")
        result = []
        for token in text.split():
            if token not in self.vocabulary:
                self.vocabulary[token] = len(self.vocabulary) + 10
            result.append(self.vocabulary[token])
        return result


class SFTTests(unittest.TestCase):
    def test_canonical_llama2_chat_prefix(self) -> None:
        prompt = format_llama2_chat_prompt("You are a predictor.", "Observe the vehicle.")
        self.assertEqual(
            prompt,
            "<s>[INST] <<SYS>>\nYou are a predictor.\n<</SYS>>\n\n"
            "Observe the vehicle. [/INST]",
        )
        self.assertNotIn("Final answer:", prompt)

    def test_only_answer_tokens_contribute_to_loss(self) -> None:
        tokenizer = FakeTokenizer()
        record = SFTRecord("s1", "system contract", "observations only", "Thought then answer")
        encoded = encode_sft_record(tokenizer, record, max_length=128)
        labels = encoded["labels"]
        input_ids = encoded["input_ids"]
        first_supervised = next(i for i, value in enumerate(labels) if value != IGNORE_INDEX)
        self.assertTrue(all(value == IGNORE_INDEX for value in labels[:first_supervised]))
        self.assertEqual(labels[first_supervised:], input_ids[first_supervised:])
        self.assertEqual(labels[-1], tokenizer.eos_token_id)

    def test_dataset_tokenization_is_lazy(self) -> None:
        tokenizer = FakeTokenizer()
        dataset = TokenizedSFTDataset(
            [SFTRecord("s1", "system", "observation", "answer")],
            tokenizer,
            max_length=64,
        )
        self.assertEqual(tokenizer.encode_calls, 0)
        self.assertEqual(len(dataset), 1)
        _ = dataset[0]
        self.assertEqual(tokenizer.encode_calls, 2)

    def test_long_prompt_is_shortened_but_answer_is_preserved(self) -> None:
        tokenizer = FakeTokenizer()
        record = SFTRecord("s1", " ".join(["system"] * 30), " ".join(["obs"] * 60), "answer")
        encoded = encode_sft_record(tokenizer, record, max_length=20)
        self.assertEqual(len(encoded["input_ids"]), 20)
        supervised = [item for item in encoded["labels"] if item != IGNORE_INDEX]
        self.assertEqual(len(supervised), 2)  # answer token plus EOS

    def test_jsonl_loader_rejects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records.jsonl"
            rows = [
                {"sample_id": "same", "system_prompt": "s", "user_prompt": "u", "answer": "a"},
                {"sample_id": "same", "system_prompt": "s", "user_prompt": "u", "answer": "b"},
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
                load_jsonl_records(path)

    def test_collator_masks_padding(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")
        tokenizer = FakeTokenizer()
        first = encode_sft_record(
            tokenizer,
            SFTRecord("a", "system", "short", "answer"),
            max_length=64,
        )
        second = encode_sft_record(
            tokenizer,
            SFTRecord("b", "system", "a longer observation", "answer two"),
            max_length=64,
        )
        features = [
            {key: value for key, value in item.items() if key != "sample_id"}
            for item in (first, second)
        ]
        batch = AnswerOnlyDataCollator(tokenizer, pad_to_multiple_of=8)(features)
        self.assertEqual(batch["input_ids"].shape[1] % 8, 0)
        padding = batch["attention_mask"] == 0
        self.assertTrue((batch["labels"][padding] == IGNORE_INDEX).all().item())


if __name__ == "__main__":
    unittest.main()
