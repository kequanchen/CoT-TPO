"""Build auditable LC-LLM JSONL records without bundling trajectory data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Union

from .config import (
    DEFAULT_LABEL_CONFIG,
    DEFAULT_PROMPT_CONFIG,
    DEFAULT_ROAD_CONFIG,
    LabelConfig,
    PromptConfig,
    RoadConfig,
)
from .cot_labels import derive_training_labels
from .data_adapter import ObservationSample, TrajectorySample
from .prompt import build_system_prompt, build_user_prompt, format_llama2_prompt
from .schema import format_lc_llm_output


def _split_name(source_split: str) -> str:
    normalized = str(source_split).strip().lower()
    if normalized not in {"train", "validation", "test"}:
        raise ValueError("source_split must be 'train', 'validation', or 'test'")
    return normalized


def _prompt_fields(
    observation: ObservationSample,
    source_split: str,
    road_config: RoadConfig,
    prompt_config: PromptConfig,
) -> dict[str, Any]:
    if not isinstance(observation, ObservationSample):
        raise TypeError("prompt records require ObservationSample")
    system = build_system_prompt(observation.layout)
    user = build_user_prompt(observation, road_config, prompt_config)
    return {
        "sample_id": observation.sample_key,
        "event_id": observation.event_key,
        "source_split": _split_name(source_split),
        "system_prompt": system,
        "user_prompt": user,
        # Deliberately excludes the assistant answer.  SFT code tokenizes the
        # answer separately so prompt tokens can be masked from the loss.
        "prompt_text": format_llama2_prompt(system, user),
    }


def build_inference_record(
    observation: ObservationSample,
    *,
    source_split: str = "test",
    road_config: RoadConfig = DEFAULT_ROAD_CONFIG,
    prompt_config: PromptConfig = DEFAULT_PROMPT_CONFIG,
) -> dict[str, Any]:
    """Create a prompt-only record with no label or future fields."""

    return _prompt_fields(observation, source_split, road_config, prompt_config)


def build_supervised_record(
    sample: TrajectorySample,
    *,
    source_split: str = "train",
    road_config: RoadConfig = DEFAULT_ROAD_CONFIG,
    prompt_config: PromptConfig = DEFAULT_PROMPT_CONFIG,
    label_config: LabelConfig = DEFAULT_LABEL_CONFIG,
) -> dict[str, Any]:
    """Create one answer-supervised record for training or validation."""

    if not isinstance(sample, TrajectorySample):
        raise TypeError("build_supervised_record expects TrajectorySample")
    split = _split_name(source_split)
    if split not in {"train", "validation"}:
        raise ValueError(
            "supervised records may be generated only from train or validation"
        )
    if sample.future is None:
        raise ValueError("supervised record requires y_future")
    record = _prompt_fields(sample.observation, split, road_config, prompt_config)
    labels = derive_training_labels(sample, label_config)
    answer = format_lc_llm_output(
        labels.notable_features,
        labels.potential_behaviors,
        labels.intention,
        labels.future_local,
        precision=prompt_config.trajectory_precision,
    )
    record.update(
        answer=answer,
        intention=labels.intention,
        future_local=labels.future_local.tolist(),
    )
    return record


def build_records(
    samples: Sequence[TrajectorySample],
    *,
    source_split: str,
    supervised: bool,
    road_config: RoadConfig = DEFAULT_ROAD_CONFIG,
    prompt_config: PromptConfig = DEFAULT_PROMPT_CONFIG,
    label_config: LabelConfig = DEFAULT_LABEL_CONFIG,
) -> list[dict[str, Any]]:
    """Build a deterministic record list for SFT or prompt-only inference."""

    split = _split_name(source_split)
    records: list[dict[str, Any]] = []
    for sample in samples:
        if supervised:
            records.append(
                build_supervised_record(
                    sample,
                    source_split=split,
                    road_config=road_config,
                    prompt_config=prompt_config,
                    label_config=label_config,
                )
            )
        else:
            records.append(
                build_inference_record(
                    sample.observation,
                    source_split=split,
                    road_config=road_config,
                    prompt_config=prompt_config,
                )
            )
    return records


def write_jsonl(
    records: Iterable[Mapping[str, Any]],
    path: Union[str, Path],
    *,
    overwrite: bool = False,
) -> Path:
    """Write records atomically enough for a local single-process workflow."""

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    materialized = [dict(record) for record in records]
    seen: set[str] = set()
    for record in materialized:
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("every record needs a non-empty sample_id")
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in materialized:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination
