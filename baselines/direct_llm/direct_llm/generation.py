"""Leakage-safe, restartable Direct LLM prediction orchestration."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .client import CompletionResult, DirectLLMAPIError


_FORBIDDEN_GROUND_TRUTH_FIELDS = frozenset(
    {
        "answer",
        "future",
        "future_trajectory",
        "ground_truth",
        "reference_trajectory",
        "target_trajectory",
        "trajectory",
        "y_future",
    }
)


@dataclass(frozen=True)
class PromptRecord:
    sample_id: str
    system_prompt: str
    user_prompt: str


def format_reminder(expected_points: int) -> str:
    """Return the fixed, ground-truth-free correction instruction."""

    if expected_points <= 0:
        raise ValueError("expected_points must be positive")
    return (
        "Your previous response did not match the required output format. "
        "Re-read the original instructions and return only one valid JSON object with "
        'exactly one key named "future_trajectory". Its value must contain exactly '
        f"{expected_points} finite numeric [x, y] coordinate pairs. Do not add markdown, "
        "commentary, units, or any other keys."
    )


def load_prompt_records(
    path: str | Path,
    *,
    max_records: int | None = None,
) -> list[PromptRecord]:
    """Load observation-only prompts and reject duplicate IDs or GT fields."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Direct LLM prompt JSONL not found: {source}")
    if max_records is not None and max_records <= 0:
        raise ValueError("max_records must be positive")
    records: list[PromptRecord] = []
    seen: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                record = prompt_record_from_mapping(payload)
            except Exception as exc:
                raise ValueError(f"invalid prompt at {source}:{line_number}: {exc}") from exc
            if record.sample_id in seen:
                raise ValueError(f"duplicate sample_id at {source}:{line_number}: {record.sample_id}")
            seen.add(record.sample_id)
            records.append(record)
            if max_records is not None and len(records) >= max_records:
                break
    if not records:
        raise ValueError(f"no prompt records found in {source}")
    return records


def prompt_record_from_mapping(payload: Mapping[str, Any]) -> PromptRecord:
    if not isinstance(payload, Mapping):
        raise TypeError("each prompt record must be a JSON object")
    forbidden = sorted(str(key) for key in payload if str(key).casefold() in _FORBIDDEN_GROUND_TRUTH_FIELDS)
    if forbidden:
        raise ValueError(
            "inference prompt records must not contain ground-truth fields: "
            + ", ".join(forbidden)
        )
    return PromptRecord(
        sample_id=_required_text(payload.get("sample_id"), "sample_id"),
        system_prompt=_required_text(payload.get("system_prompt"), "system_prompt"),
        user_prompt=_required_text(payload.get("user_prompt"), "user_prompt"),
    )


def run_predictions(
    records: Iterable[PromptRecord],
    output_path: str | Path,
    completion: Callable[[PromptRecord, str | None], CompletionResult],
    parser: Callable[[str], Any],
    *,
    expected_points: int,
    max_format_retries: int,
    resume: bool = True,
    overwrite: bool = False,
) -> dict[str, int]:
    """Call the API one sample at a time and checkpoint every JSONL row."""

    if expected_points <= 0:
        raise ValueError("expected_points must be positive")
    if max_format_retries < 0:
        raise ValueError("max_format_retries cannot be negative")
    if resume and overwrite:
        raise ValueError("resume and overwrite cannot both be enabled")
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not resume and not overwrite:
        raise FileExistsError(
            f"prediction output exists: {destination}; use resume or overwrite"
        )
    completed = _existing_sample_ids(destination) if resume else set()
    normalized = list(records)
    _reject_duplicate_input_ids(normalized)
    pending = [record for record in normalized if record.sample_id not in completed]
    counts = {
        "input": len(normalized),
        "skipped": len(normalized) - len(pending),
        "written": 0,
        "ok": 0,
        "parse_errors": 0,
        "api_errors": 0,
    }
    mode = "a" if resume and destination.exists() else "w"
    reminder = format_reminder(expected_points)
    with destination.open(mode, encoding="utf-8") as handle:
        for record in pending:
            row = _predict_one(
                record,
                completion,
                parser,
                reminder=reminder,
                max_format_retries=max_format_retries,
            )
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            counts["written"] += 1
            if row["status"] == "ok":
                counts["ok"] += 1
            elif row["status"] == "parse_error":
                counts["parse_errors"] += 1
            else:
                counts["api_errors"] += 1
    return counts


def _predict_one(
    record: PromptRecord,
    completion: Callable[[PromptRecord, str | None], CompletionResult],
    parser: Callable[[str], Any],
    *,
    reminder: str,
    max_format_retries: int,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    last_raw_output = ""
    for format_attempt in range(max_format_retries + 1):
        correction = None if format_attempt == 0 else reminder
        try:
            result = completion(record, correction)
            if not isinstance(result, CompletionResult):
                raise TypeError("completion callback must return CompletionResult")
        except DirectLLMAPIError as exc:
            safe_error = str(exc)
            attempts.append(
                {
                    "format_attempt": format_attempt,
                    "api_attempts": exc.attempts,
                    "status": "api_error",
                    "error": safe_error,
                }
            )
            return {
                "sample_id": record.sample_id,
                "status": "api_error",
                "raw_output": last_raw_output,
                "error": safe_error,
                "attempts": attempts,
            }
        except Exception as exc:
            safe_error = f"unexpected client failure: {type(exc).__name__}"
            attempts.append(
                {
                    "format_attempt": format_attempt,
                    "api_attempts": 0,
                    "status": "api_error",
                    "error": safe_error,
                }
            )
            return {
                "sample_id": record.sample_id,
                "status": "api_error",
                "raw_output": last_raw_output,
                "error": safe_error,
                "attempts": attempts,
            }

        last_raw_output = result.text
        try:
            payload = _parsed_payload(parser(result.text))
        except Exception as exc:
            parse_error = _safe_parse_error(exc)
            attempts.append(
                {
                    "format_attempt": format_attempt,
                    "api_attempts": result.api_attempts,
                    "status": "parse_error",
                    "raw_output": result.text,
                    "error": parse_error,
                }
            )
            if format_attempt < max_format_retries:
                continue
            return {
                "sample_id": record.sample_id,
                "status": "parse_error",
                "raw_output": result.text,
                "error": parse_error,
                "attempts": attempts,
            }

        attempts.append(
            {
                "format_attempt": format_attempt,
                "api_attempts": result.api_attempts,
                "status": "ok",
                "raw_output": result.text,
            }
        )
        return {
            "sample_id": record.sample_id,
            "status": "ok",
            "raw_output": result.text,
            "attempts": attempts,
            **payload,
        }
    raise AssertionError("unreachable format retry state")


def _parsed_payload(parsed: Any) -> dict[str, Any]:
    if hasattr(parsed, "as_json_dict"):
        parsed = parsed.as_json_dict()
    elif is_dataclass(parsed):
        parsed = asdict(parsed)
    elif isinstance(parsed, Mapping):
        parsed = dict(parsed)
    elif hasattr(parsed, "trajectory"):
        parsed = {"trajectory": getattr(parsed, "trajectory")}
    elif hasattr(parsed, "tolist"):
        parsed = {"trajectory": parsed}
    else:
        raise TypeError("strict parser returned an unsupported object")
    if not isinstance(parsed, Mapping):
        raise TypeError("strict parser must return a mapping-like value")
    payload = {str(key): _json_compatible(value) for key, value in parsed.items()}
    trajectory = payload.pop("trajectory", None)
    future_trajectory = payload.pop("future_trajectory", None)
    if trajectory is None:
        trajectory = future_trajectory
    if trajectory is None:
        raise ValueError("strict parser result does not contain trajectory")
    for protected in ("sample_id", "status", "raw_output", "attempts", "error"):
        payload.pop(protected, None)
    payload["trajectory"] = trajectory
    json.dumps(payload, ensure_ascii=False, allow_nan=False)
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
    raise TypeError(f"parsed output contains non-JSON value: {type(value).__name__}")


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


def _reject_duplicate_input_ids(records: list[PromptRecord]) -> None:
    seen: set[str] = set()
    for record in records:
        if record.sample_id in seen:
            raise ValueError(f"duplicate input sample_id: {record.sample_id}")
        seen.add(record.sample_id)


def _safe_parse_error(exc: Exception) -> str:
    detail = " ".join(str(exc).split())[:300]
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()
