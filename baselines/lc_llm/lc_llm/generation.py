"""Batched and resumable generation for the adapted LC-LLM baseline."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .sft import SFTRecord, format_llama2_chat_prompt


@dataclass(frozen=True)
class GenerationSettings:
    batch_size: int = 4
    max_input_tokens: int = 3072
    max_new_tokens: int = 1024
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    num_beams: int = 1

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("generation batch_size must be positive")
        if self.max_input_tokens <= 0 or self.max_new_tokens <= 0:
            raise ValueError("generation token limits must be positive")
        if self.do_sample and self.temperature <= 0.0:
            raise ValueError("temperature must be positive when sampling")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.num_beams <= 0:
            raise ValueError("num_beams must be positive")
        if self.do_sample and self.num_beams != 1:
            raise ValueError("this baseline does not combine sampling with beam search")

    def validate_context_window(self, max_sequence_length: int) -> None:
        """Reject a prompt/output budget that exceeds the model context."""

        self.validate()
        if max_sequence_length <= 0:
            raise ValueError("model max_sequence_length must be positive")
        requested = self.max_input_tokens + self.max_new_tokens
        if requested > max_sequence_length:
            raise ValueError(
                "generation token budget exceeds model.max_seq_length: "
                f"{self.max_input_tokens} + {self.max_new_tokens} > "
                f"{max_sequence_length}"
            )


def prompt_for_inference(record: SFTRecord) -> str:
    """Build an inference prefix without ever including ``record.answer``."""

    return format_llama2_chat_prompt(record.system_prompt, record.user_prompt)


def tokenize_generation_prompts(
    tokenizer: Any,
    records: Sequence[SFTRecord],
    *,
    max_input_tokens: int,
) -> list[list[int]]:
    """Tokenize prompts with an explicit, auditable head-and-tail policy.

    Long prompts retain their system-contract prefix and the latest observation
    plus closing ``[/INST]`` suffix.  This avoids the tokenizer default of
    silently right-truncating the latest observation and assistant delimiter.
    """

    if max_input_tokens < 8:
        raise ValueError("max_input_tokens must be at least 8")
    encoded_prompts: list[list[int]] = []
    for record in records:
        text = prompt_for_inference(record)
        if hasattr(tokenizer, "encode"):
            token_ids = tokenizer.encode(text, add_special_tokens=False)
        else:
            token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if token_ids and isinstance(token_ids[0], list):
            if len(token_ids) != 1:
                raise ValueError("tokenizer unexpectedly returned a batched prompt encoding")
            token_ids = token_ids[0]
        normalized = [int(item) for item in token_ids]
        if not normalized:
            raise ValueError(f"tokenized prompt is empty for {record.sample_id}")
        if len(normalized) > max_input_tokens:
            prefix = (max_input_tokens + 1) // 2
            suffix = max_input_tokens - prefix
            normalized = normalized[:prefix] + normalized[-suffix:]
        encoded_prompts.append(normalized)
    return encoded_prompts


def generate_text_batch(
    model: Any,
    tokenizer: Any,
    records: Sequence[SFTRecord],
    settings: GenerationSettings,
) -> list[str]:
    """Generate assistant continuations for one batch of prompt records."""

    settings.validate()
    if not records:
        return []
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for LC-LLM generation") from exc

    tokenized = tokenize_generation_prompts(
        tokenizer,
        records,
        max_input_tokens=settings.max_input_tokens,
    )
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token_id")
    input_width = max(len(item) for item in tokenized)
    input_rows: list[list[int]] = []
    attention_rows: list[list[int]] = []
    for token_ids in tokenized:
        padding = input_width - len(token_ids)
        input_rows.append([int(pad_token_id)] * padding + token_ids)
        attention_rows.append([0] * padding + [1] * len(token_ids))
    encoded = {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
    }
    device = _model_input_device(model)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": settings.max_new_tokens,
        "do_sample": settings.do_sample,
        "num_beams": settings.num_beams,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if settings.do_sample:
        generation_kwargs.update(temperature=settings.temperature, top_p=settings.top_p)
    with torch.inference_mode():
        output_ids = model.generate(**encoded, **generation_kwargs)
    continuations = output_ids[:, input_width:]
    texts = tokenizer.batch_decode(
        continuations,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if len(texts) != len(records):
        raise RuntimeError("model returned a different number of generations than prompts")
    return [text.strip() for text in texts]


def run_batched_prediction(
    records: Iterable[SFTRecord],
    output_path: str | Path,
    predictor: Callable[[Sequence[SFTRecord]], Sequence[str]],
    *,
    batch_size: int,
    resume: bool = True,
    overwrite: bool = False,
    parser: Callable[[str], Mapping[str, Any] | Any] | None = None,
    fail_fast: bool = False,
) -> dict[str, int]:
    """Generate JSONL predictions with durable batch checkpoints.

    Existing sample identifiers are skipped in resume mode, including records
    that previously failed.  This prevents duplicate identifiers; delete or
    move a failed output file before rerunning if regeneration is desired.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if resume and overwrite:
        raise ValueError("resume and overwrite cannot both be enabled")
    if destination.exists() and not resume and not overwrite:
        raise FileExistsError(
            f"prediction output already exists: {destination}; use resume or overwrite"
        )
    completed = _existing_sample_ids(destination) if resume else set()
    normalized = list(records)
    _reject_duplicate_input_ids(normalized)
    pending = [record for record in normalized if record.sample_id not in completed]
    counters = {
        "input": len(normalized),
        "skipped": len(normalized) - len(pending),
        "written": 0,
        "ok": 0,
        "errors": 0,
    }
    mode = "a" if resume and destination.exists() else "w"
    with destination.open(mode, encoding="utf-8") as handle:
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            rows: list[dict[str, Any]] = []
            try:
                outputs = list(predictor(batch))
                if len(outputs) != len(batch):
                    raise RuntimeError("predictor returned a different number of outputs")
                for record, raw_output in zip(batch, outputs):
                    if not isinstance(raw_output, str) or not raw_output.strip():
                        raise ValueError(f"empty model output for {record.sample_id}")
                    row: dict[str, Any] = {
                        "sample_id": record.sample_id,
                        "status": "ok",
                        "raw_output": raw_output.strip(),
                    }
                    if parser is not None:
                        try:
                            row.update(_parsed_payload(parser(raw_output)))
                        except Exception as exc:
                            row["status"] = "parse_error"
                            row["error"] = f"{type(exc).__name__}: {exc}"
                    rows.append(row)
            except Exception as exc:
                if fail_fast:
                    raise
                rows = [
                    {
                        "sample_id": record.sample_id,
                        "status": "generation_error",
                        "raw_output": "",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    for record in batch
                ]
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                counters["written"] += 1
                if row["status"] == "ok":
                    counters["ok"] += 1
                else:
                    counters["errors"] += 1
            # Each completed batch is a restart boundary.
            handle.flush()
            os.fsync(handle.fileno())
    return counters


def _parsed_payload(parsed: Mapping[str, Any] | Any) -> dict[str, Any]:
    if hasattr(parsed, "as_json_dict"):
        parsed = parsed.as_json_dict()
    elif is_dataclass(parsed):
        parsed = asdict(parsed)
    elif not isinstance(parsed, Mapping) and hasattr(parsed, "__dict__"):
        parsed = vars(parsed)
    if not isinstance(parsed, Mapping):
        raise TypeError("prediction parser must return a mapping or dataclass-like object")
    payload = {str(key): _json_compatible(value) for key, value in parsed.items()}
    # Avoid allowing a parser to overwrite provenance/status fields.
    for protected in ("sample_id", "status", "raw_output", "error"):
        payload.pop(protected, None)
    json.dumps(payload, ensure_ascii=False)
    return payload


def _json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_compatible(value.tolist())
    raise TypeError(f"parsed output contains a non-JSON value: {type(value).__name__}")


def _existing_sample_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    sample_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                sample_id = payload["sample_id"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"invalid existing output at {path}:{line_number}") from exc
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"invalid sample_id at {path}:{line_number}")
            if sample_id in sample_ids:
                raise ValueError(f"duplicate sample_id in existing output: {sample_id}")
            sample_ids.add(sample_id)
    return sample_ids


def _reject_duplicate_input_ids(records: Sequence[SFTRecord]) -> None:
    seen: set[str] = set()
    for record in records:
        if record.sample_id in seen:
            raise ValueError(f"duplicate input sample_id: {record.sample_id}")
        seen.add(record.sample_id)


def _model_input_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration) as exc:
        raise ValueError("cannot determine model input device") from exc
