"""Causal-language-model supervision utilities for the adapted LC-LLM baseline.

The helpers in this module intentionally depend only on the Python standard
library.  ``torch`` is imported only when a batch is collated, so data and
prompt validation remain usable in lightweight environments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


IGNORE_INDEX = -100


@dataclass(frozen=True)
class SFTRecord:
    """One supervised LC-LLM conversation.

    ``answer`` contains the paper-style joint reasoning, intention, and future
    trajectory response.  It is never included in an inference prompt.
    """

    sample_id: str
    system_prompt: str
    user_prompt: str
    answer: str
    event_id: str | None = None
    source_split: str | None = None


def format_llama2_chat_prompt(system_prompt: str, user_prompt: str) -> str:
    """Return the canonical single-turn Llama-2-chat instruction prefix.

    The returned text ends immediately after ``[/INST]``.  Training code adds
    the assistant answer as a separate token segment, which makes answer-only
    loss masking explicit and auditable.
    """

    system = _nonempty_text(system_prompt, "system_prompt")
    user = _nonempty_text(user_prompt, "user_prompt")
    return f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"


def record_from_mapping(record: Mapping[str, Any], *, require_answer: bool = True) -> SFTRecord:
    """Validate a JSON-compatible record and normalize it to :class:`SFTRecord`."""

    if not isinstance(record, Mapping):
        raise TypeError("each LC-LLM record must be a JSON object")
    sample_id = _nonempty_text(record.get("sample_id"), "sample_id")
    system_prompt = _nonempty_text(record.get("system_prompt"), "system_prompt")
    user_prompt = _nonempty_text(record.get("user_prompt"), "user_prompt")
    raw_answer = record.get("answer", "")
    if require_answer:
        answer = _nonempty_text(raw_answer, "answer")
    elif raw_answer is None:
        answer = ""
    elif isinstance(raw_answer, str):
        answer = raw_answer.strip()
    else:
        raise TypeError("answer must be a string when provided")
    event_id = _optional_nonempty_text(record.get("event_id"), "event_id")
    source_split = _optional_nonempty_text(record.get("source_split"), "source_split")
    if source_split is not None:
        source_split = source_split.lower()
        if source_split not in {"train", "validation", "test"}:
            raise ValueError("source_split must be 'train', 'validation', or 'test'")
    return SFTRecord(
        sample_id=sample_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        answer=answer,
        event_id=event_id,
        source_split=source_split,
    )


def load_jsonl_records(
    path: str | Path,
    *,
    require_answer: bool = True,
    max_records: int | None = None,
    expected_source_split: str | None = None,
    require_event_id: bool = False,
) -> list[SFTRecord]:
    """Load validated records while rejecting duplicate or misassigned examples."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"LC-LLM JSONL file not found: {source}")
    if max_records is not None and max_records <= 0:
        raise ValueError("max_records must be positive")
    expected_split = None
    if expected_source_split is not None:
        expected_split = str(expected_source_split).strip().lower()
        if expected_split not in {"train", "validation", "test"}:
            raise ValueError(
                "expected_source_split must be 'train', 'validation', or 'test'"
            )
    records: list[SFTRecord] = []
    seen: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                record = record_from_mapping(payload, require_answer=require_answer)
            except Exception as exc:
                raise ValueError(f"invalid record at {source}:{line_number}: {exc}") from exc
            if record.sample_id in seen:
                raise ValueError(f"duplicate sample_id at {source}:{line_number}: {record.sample_id}")
            if expected_split is not None and record.source_split != expected_split:
                raise ValueError(
                    f"source_split at {source}:{line_number} is {record.source_split!r}; "
                    f"expected {expected_split!r}"
                )
            if require_event_id and record.event_id is None:
                raise ValueError(f"event_id is required at {source}:{line_number}")
            seen.add(record.sample_id)
            records.append(record)
            if max_records is not None and len(records) >= max_records:
                break
    if not records:
        raise ValueError(f"no records found in {source}")
    return records


def encode_sft_record(
    tokenizer: Any,
    record: SFTRecord | Mapping[str, Any],
    *,
    max_length: int,
) -> dict[str, list[int] | str]:
    """Tokenize one example and mask every non-answer label.

    If the conversation is too long, only the prompt is shortened, retaining
    both its beginning (system contract) and end (latest observation and
    ``[/INST]`` delimiter).  Answers are never silently removed; users must
    increase ``max_length`` if an answer alone cannot fit.
    """

    if max_length < 16:
        raise ValueError("max_length must be at least 16")
    normalized = (
        record
        if isinstance(record, SFTRecord)
        else record_from_mapping(record, require_answer=True)
    )
    prompt = format_llama2_chat_prompt(normalized.system_prompt, normalized.user_prompt)
    prompt_ids = _encode_text(tokenizer, prompt)
    answer_ids = _encode_text(tokenizer, " " + normalized.answer.strip())
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and (not answer_ids or answer_ids[-1] != int(eos_token_id)):
        answer_ids.append(int(eos_token_id))
    if not answer_ids:
        raise ValueError("the tokenized answer is empty")
    if len(answer_ids) >= max_length:
        raise ValueError(
            f"answer for {normalized.sample_id!r} has {len(answer_ids)} tokens and cannot "
            f"fit max_length={max_length}; increase model.max_seq_length"
        )
    prompt_budget = max_length - len(answer_ids)
    if len(prompt_ids) > prompt_budget:
        prompt_ids = _retain_prompt_ends(prompt_ids, prompt_budget)
    input_ids = prompt_ids + answer_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids.copy()
    return {
        "sample_id": normalized.sample_id,
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


class TokenizedSFTDataset:
    """Lazy framework-neutral dataset accepted directly by HF ``Trainer``.

    Only normalized records are retained.  Tokenization happens in
    ``__getitem__`` so a full 32k-sample, long-context corpus is not duplicated
    as several large Python integer lists in host memory.
    """

    def __init__(
        self,
        records: Iterable[SFTRecord | Mapping[str, Any]],
        tokenizer: Any,
        *,
        max_length: int,
    ) -> None:
        self._records = [
            record
            if isinstance(record, SFTRecord)
            else record_from_mapping(record, require_answer=True)
            for record in records
        ]
        if not self._records:
            raise ValueError("SFT dataset cannot be empty")
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Trainer/model inputs must not include metadata strings.
        example = encode_sft_record(
            self._tokenizer,
            self._records[index],
            max_length=self._max_length,
        )
        return {key: value for key, value in example.items() if key != "sample_id"}


class AnswerOnlyDataCollator:
    """Right-pad causal-LM batches while retaining answer-only label masks."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        pad_to_multiple_of: int | None = 8,
    ) -> None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise ValueError("tokenizer must define pad_token_id or eos_token_id")
        if pad_to_multiple_of is not None and pad_to_multiple_of <= 0:
            raise ValueError("pad_to_multiple_of must be positive")
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: Sequence[Mapping[str, Sequence[int]]]) -> dict[str, Any]:
        if not features:
            raise ValueError("cannot collate an empty batch")
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - exercised only in minimal installs
            raise RuntimeError("PyTorch is required to collate SFT batches") from exc
        max_length = max(len(item["input_ids"]) for item in features)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            max_length = ((max_length + multiple - 1) // multiple) * multiple
        batch_ids: list[list[int]] = []
        batch_masks: list[list[int]] = []
        batch_labels: list[list[int]] = []
        for item in features:
            length = len(item["input_ids"])
            if len(item["attention_mask"]) != length or len(item["labels"]) != length:
                raise ValueError("input_ids, attention_mask, and labels must have equal lengths")
            padding = max_length - length
            batch_ids.append(list(item["input_ids"]) + [self.pad_token_id] * padding)
            batch_masks.append(list(item["attention_mask"]) + [0] * padding)
            batch_labels.append(list(item["labels"]) + [IGNORE_INDEX] * padding)
        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_masks, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        encoded = tokenizer.encode(text, add_special_tokens=False)
    else:
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("tokenizer unexpectedly returned a batched encoding")
        encoded = encoded[0]
    return [int(token) for token in encoded]


def _retain_prompt_ends(tokens: Sequence[int], budget: int) -> list[int]:
    if budget <= 0:
        raise ValueError("prompt token budget must be positive")
    if len(tokens) <= budget:
        return list(tokens)
    prefix = (budget + 1) // 2
    suffix = budget - prefix
    if suffix == 0:
        return list(tokens[:prefix])
    return list(tokens[:prefix]) + list(tokens[-suffix:])


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_nonempty_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _nonempty_text(value, field)
