"""Build prompt-only Direct LLM JSONL records without ground truth."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Union

from .config import DEFAULT_PROMPT_CONFIG, PromptConfig
from .data_adapter import ObservationSample
from .prompt import build_system_prompt, build_user_prompt, format_prompt_text


_FORBIDDEN_PROMPT_FIELDS = {
    "answer",
    "future",
    "future_local",
    "y_future",
    "lane_status",
    "time_since_crossing",
    "intention",
}


def build_inference_record(
    observation: ObservationSample,
    *,
    source_split: str = "test",
    prompt_config: PromptConfig = DEFAULT_PROMPT_CONFIG,
) -> dict[str, Any]:
    """Create one observation-only prompt record."""

    if not isinstance(observation, ObservationSample):
        raise TypeError("build_inference_record accepts ObservationSample only")
    if str(source_split).strip().lower() != "test":
        raise ValueError("Direct LLM prompt records are restricted to the test split")
    system = build_system_prompt(observation.layout)
    user = build_user_prompt(observation, prompt_config)
    record = {
        "sample_id": observation.sample_id,
        "event_id": observation.event_key,
        "source_split": "test",
        "system_prompt": system,
        "user_prompt": user,
        "prompt_text": format_prompt_text(system, user),
    }
    _assert_prompt_record_safe(record)
    return record


def build_records(
    observations: Sequence[ObservationSample],
    *,
    source_split: str = "test",
    prompt_config: PromptConfig = DEFAULT_PROMPT_CONFIG,
) -> list[dict[str, Any]]:
    """Build deterministic test prompts; no supervised mode exists here."""

    records = [
        build_inference_record(
            observation,
            source_split=source_split,
            prompt_config=prompt_config,
        )
        for observation in observations
    ]
    _validate_unique_ids(records)
    return records


def _assert_prompt_record_safe(record: Mapping[str, Any]) -> None:
    leaked = _FORBIDDEN_PROMPT_FIELDS.intersection(record)
    if leaked:
        raise ValueError(f"prompt record contains forbidden label fields: {sorted(leaked)}")
    expected = {
        "sample_id",
        "event_id",
        "source_split",
        "system_prompt",
        "user_prompt",
        "prompt_text",
    }
    if set(record) != expected:
        raise ValueError(f"unexpected prompt record fields: {sorted(set(record) - expected)}")


def _validate_unique_ids(records: Iterable[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    for record in records:
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("every prompt record needs a non-empty sample_id")
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)


def write_jsonl(
    records: Iterable[Mapping[str, Any]],
    path: Union[str, Path],
    *,
    overwrite: bool = False,
) -> Path:
    """Validate and atomically write data-free prompt records."""

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {destination}")
    materialized = [dict(record) for record in records]
    for record in materialized:
        _assert_prompt_record_safe(record)
    _validate_unique_ids(materialized)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in materialized:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination
