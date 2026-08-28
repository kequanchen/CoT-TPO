"""Parse LLM CoT JSON responses into fixed-dimensional strategy vectors.

This script converts the LLM teacher outputs produced by ``lane_change_cot.py``
into the ``ids.npy`` and ``c.npy`` files consumed by the MLP student and the
downstream CoT-TP model.

The public vector schema follows the manuscript appendix:

    c = [
        phase_onehot(3),
        strategy_scores(12),
        speed_change_onehot(3),
        continuous_values(6),
        continuous_masks(6),
    ]

The resulting vector dimension is 30.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np


PHASE2STATUS = {"anticipation": 0, "crossing": 1, "relaxation": 2}

STRATEGY_POOL = [
    "DECISIVE_MERGE",
    "PROBING_APPROACH",
    "ACCELERATE_PASS",
    "DECELERATE_THEN_MERGE",
    "MAINTAIN_AND_OBSERVE",
    "YIELD_FOR_GAP2",
    "ACCELERATE_STABILIZE",
    "DECELERATE_STABILIZE",
    "SPEED_MATCH",
    "MAINTAIN_STABLE",
    "HOLD_AND_WAIT",
    "DECELERATE_AVOID_CRASH",
]

SPEED_CHANGE_VOCAB = ["ACCEL", "KEEP", "DECEL"]

CONT_FIELDS = [
    ("aggressiveness", ("decision", "aggressiveness")),
    ("risk_tolerance", ("decision", "risk_tolerance")),
    ("lateral_intent", ("decision", "lateral_intent")),
    ("longitudinal_intent", ("decision", "longitudinal_intent")),
    ("confidence", ("decision", "confidence")),
    ("stability_level", ("decision", "stability_level")),
]

FLOAT_PATTERN = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?")
SAMPLE_ID_PATTERN = re.compile(r"response_sample_(\d+)\.txt$")


def extract_first_json(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object from a response file."""

    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found")

    depth = 0
    in_string = False
    escaped = False
    for pos, char in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : pos + 1])

    raise ValueError("Unclosed JSON object")


def sample_id_from_response_path(path: Path) -> int:
    match = SAMPLE_ID_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Cannot infer sample id from {path.name}")
    return int(match.group(1))


def normalize_phase(value: Any, fallback: str = "anticipation") -> str:
    text = str(value or fallback).strip().lower()
    if "anticip" in text:
        return "anticipation"
    if "cross" in text:
        return "crossing"
    if "relax" in text or "stabil" in text:
        return "relaxation"
    return fallback.lower()


def parse_phase_from_prompt(prompt_text: str) -> Optional[str]:
    phase_match = re.search(r"Phase\s*:\s*([A-Za-z_ -]+)", prompt_text, flags=re.IGNORECASE)
    if phase_match:
        return normalize_phase(phase_match.group(1))
    lower = prompt_text.lower()
    for phase in PHASE2STATUS:
        if phase in lower:
            return phase
    return None


def normalize_strategy(value: Any) -> str:
    text = str(value or "NONE").strip().upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text).strip("_")
    aliases = {
        "DECISIVE": "DECISIVE_MERGE",
        "PROBING": "PROBING_APPROACH",
        "ACCELERATE": "ACCELERATE_PASS",
        "DECELERATE": "DECELERATE_THEN_MERGE",
        "MAINTAIN": "MAINTAIN_AND_OBSERVE",
        "YIELD": "YIELD_FOR_GAP2",
    }
    return aliases.get(text, text)


def normalize_speed_change(value: Any) -> str:
    text = str(value or "KEEP").strip().upper()
    if "DECEL" in text or "SLOW" in text or "BRAKE" in text:
        return "DECEL"
    if "ACCEL" in text or "SPEED_UP" in text or "SPEED UP" in text or "CATCH" in text:
        return "ACCEL"
    return "KEEP"


def normalize_interaction(value: Any) -> str:
    text = str(value or "NONE").strip().upper()
    return text if text in {"OL", "OF", "TL", "TF", "TFF", "NONE"} else "NONE"


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float, np.number)):
        out = float(value)
        return out if np.isfinite(out) else default

    match = FLOAT_PATTERN.search(str(value))
    if not match:
        return default
    out = float(match.group(0))
    return out if np.isfinite(out) else default


def one_hot(value: str, vocab: list[str]) -> np.ndarray:
    arr = np.zeros(len(vocab), dtype=np.float32)
    if value in vocab:
        arr[vocab.index(value)] = 1.0
    return arr


def get_nested(payload: dict[str, Any], path: tuple[str, str]) -> Any:
    section, key = path
    obj = payload.get(section, {}) or {}
    if not isinstance(obj, dict):
        return None
    return obj.get(key)


def build_strategy_vector(payload: dict[str, Any], fallback_phase: str) -> tuple[np.ndarray, dict[str, Any]]:
    phase = normalize_phase(payload.get("phase"), fallback=fallback_phase)
    status = PHASE2STATUS.get(phase, -1)
    phase_oh = np.zeros(3, dtype=np.float32)
    if status >= 0:
        phase_oh[status] = 1.0

    decision = payload.get("decision", {}) or {}
    if not isinstance(decision, dict):
        decision = {}
    guidance = payload.get("prediction_guidance", {}) or {}
    if not isinstance(guidance, dict):
        guidance = {}

    raw_scores = decision.get("strategy_scores", {}) or {}
    if not isinstance(raw_scores, dict):
        raw_scores = {}

    scores = np.zeros(len(STRATEGY_POOL), dtype=np.float32)
    ignored_scores = {}
    for key, value in raw_scores.items():
        strategy = normalize_strategy(key)
        if strategy in STRATEGY_POOL:
            scores[STRATEGY_POOL.index(strategy)] = np.clip(safe_float(value), 0.0, 1.0)
        else:
            ignored_scores[str(key)] = value

    speed_change = normalize_speed_change(guidance.get("expected_speed_change"))
    speed_oh = one_hot(speed_change, SPEED_CHANGE_VOCAB)

    continuous = np.zeros(len(CONT_FIELDS), dtype=np.float32)
    continuous_mask = np.zeros(len(CONT_FIELDS), dtype=np.float32)
    for idx, (_, path) in enumerate(CONT_FIELDS):
        value = get_nested(payload, path)
        if value is not None:
            continuous[idx] = safe_float(value)
            continuous_mask[idx] = 1.0

    vector = np.concatenate([phase_oh, scores, speed_oh, continuous, continuous_mask], axis=0).astype(np.float32)

    meta = {
        "phase": phase,
        "status": int(status),
        "primary_strategy": normalize_strategy(decision.get("primary_strategy")),
        "target_gap": str(decision.get("target_gap", "NONE")),
        "speed_change_norm": speed_change,
        "key_interaction": normalize_interaction(guidance.get("key_interaction")),
        "ignored_strategy_scores": ignored_scores,
        "c_dim": int(vector.shape[0]),
    }
    return vector, meta


def iter_response_paths(responses_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/response_sample_*.txt" if recursive else "response_sample_*.txt"
    return sorted(responses_dir.glob(pattern))


def read_prompt_phase(prompts_dir: Optional[Path], sample_id: int) -> Optional[str]:
    if prompts_dir is None:
        return None
    prompt_path = prompts_dir / f"prompt_sample_{sample_id}.txt"
    if not prompt_path.exists():
        return None
    return parse_phase_from_prompt(prompt_path.read_text(encoding="utf-8", errors="ignore"))


def write_meta_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "sample_id",
        "phase",
        "status",
        "primary_strategy",
        "target_gap",
        "speed_change_norm",
        "key_interaction",
        "c_dim",
        "response_path",
        "prompt_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def write_vocab(path: Path) -> None:
    vocab = {
        "phase2status": PHASE2STATUS,
        "strategy_pool": STRATEGY_POOL,
        "speed_change_vocab": SPEED_CHANGE_VOCAB,
        "continuous_fields": [name for name, _ in CONT_FIELDS],
        "c_layout": {
            "phase_onehot": 3,
            "strategy_scores": len(STRATEGY_POOL),
            "speed_change_onehot": len(SPEED_CHANGE_VOCAB),
            "continuous": len(CONT_FIELDS),
            "continuous_mask": len(CONT_FIELDS),
        },
        "c_dim_total": 3 + len(STRATEGY_POOL) + len(SPEED_CHANGE_VOCAB) + 2 * len(CONT_FIELDS),
    }
    path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse LLM response JSON files into CoT-TP strategy vectors.")
    parser.add_argument("--responses-dir", type=Path, required=True, help="Directory containing response_sample_<id>.txt files.")
    parser.add_argument("--prompts-dir", type=Path, default=None, help="Optional directory containing prompt_sample_<id>.txt files.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--recursive", action="store_true", help="Search response files recursively.")
    parser.add_argument("--fallback-phase", type=str, default="anticipation", choices=["anticipation", "crossing", "relaxation"])
    parser.add_argument("--strict", action="store_true", help="Return a nonzero exit code if any response cannot be parsed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Remove only files derived by this parser. This prevents a failed or empty
    # parse from leaving apparently valid vectors from an earlier dataset run.
    for name in (
        "ids.npy",
        "c.npy",
        "vocab.json",
        "meta.csv",
        "records.jsonl",
        "summary.json",
        "errors.log",
    ):
        path = args.out_dir / name
        if path.exists():
            path.unlink()

    response_paths = iter_response_paths(args.responses_dir, args.recursive)
    errors = []
    ids = []
    vectors = []
    meta_rows = []

    records_path = args.out_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as records_f:
        for response_path in response_paths:
            try:
                sample_id = sample_id_from_response_path(response_path)
                response_text = response_path.read_text(encoding="utf-8", errors="ignore")
                payload = extract_first_json(response_text)
                prompt_phase = read_prompt_phase(args.prompts_dir, sample_id)
                fallback_phase = prompt_phase or args.fallback_phase
                vector, meta = build_strategy_vector(payload, fallback_phase=fallback_phase)
            except Exception as exc:
                errors.append(f"[parse_fail] path={response_path} {type(exc).__name__}: {exc}")
                continue

            ids.append(sample_id)
            vectors.append(vector)
            prompt_path = args.prompts_dir / f"prompt_sample_{sample_id}.txt" if args.prompts_dir else None
            meta_row = {
                "sample_id": sample_id,
                **meta,
                "response_path": str(response_path),
                "prompt_path": str(prompt_path) if prompt_path and prompt_path.exists() else "",
            }
            meta_rows.append(meta_row)
            records_f.write(
                json.dumps(
                    {
                        **meta_row,
                        "decision": payload.get("decision", {}),
                        "prediction_guidance": payload.get("prediction_guidance", {}),
                        "c": vector.tolist(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    if vectors:
        np.save(args.out_dir / "ids.npy", np.asarray(ids, dtype=np.int32))
        np.save(args.out_dir / "c.npy", np.stack(vectors, axis=0).astype(np.float32))
        write_meta_csv(args.out_dir / "meta.csv", meta_rows)

    write_vocab(args.out_dir / "vocab.json")

    summary = {
        "responses_dir": str(args.responses_dir),
        "prompts_dir": str(args.prompts_dir) if args.prompts_dir else None,
        "out_dir": str(args.out_dir),
        "parsed": len(ids),
        "errors": len(errors),
        "c_dim": int(vectors[0].shape[0]) if vectors else None,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        (args.out_dir / "errors.log").write_text("\n".join(errors), encoding="utf-8")

    print(f"Parsed {len(ids)} responses into {args.out_dir}")
    if vectors:
        print(f"Strategy vector dimension: {vectors[0].shape[0]}")
    if errors:
        print(f"Warnings: {len(errors)} parse errors written to {args.out_dir / 'errors.log'}")
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
