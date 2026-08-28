#!/usr/bin/env python3
"""Train the adapted LC-LLM causal LM with answer-only PEFT LoRA SFT."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping


BASELINE_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a local/Hugging Face causal LM for the adapted LC-LLM baseline."
    )
    parser.add_argument("--config", required=True, help="Path to an LC-LLM JSON configuration")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional training smoke-test limit")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="A Trainer checkpoint directory, or 'latest' to auto-detect the newest checkpoint",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override training.seed")
    return parser.parse_args()


def train(args: argparse.Namespace) -> Path:
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    config_path = Path(args.config).expanduser().resolve()
    config = _load_json_config(config_path)
    paths = _section(config, "paths")
    model_cfg = _section(config, "model")
    lora_cfg = _section(config, "lora")
    training_cfg = _section(config, "training")
    if bool(model_cfg.get("trust_remote_code", False)):
        raise ValueError("model.trust_remote_code must remain false for this public baseline")

    from lc_llm.modeling import (
        LoraSettings,
        ModelLoadSettings,
        attach_lora_for_training,
        load_causal_lm,
        load_tokenizer,
        set_global_seed,
    )
    from lc_llm.sft import AnswerOnlyDataCollator, TokenizedSFTDataset, load_jsonl_records

    try:
        from transformers import Trainer, TrainingArguments
        from transformers.trainer_utils import get_last_checkpoint
    except ImportError as exc:
        raise RuntimeError(
            "training LC-LLM requires torch, transformers, peft, accelerate, and optionally bitsandbytes"
        ) from exc

    seed = int(args.seed if args.seed is not None else training_cfg.get("seed", 42))
    set_global_seed(seed, deterministic=bool(training_cfg.get("deterministic", False)))
    train_path = _configured_path(paths.get("train_jsonl"), "paths.train_jsonl")
    validation_path = _configured_path(
        paths.get("validation_jsonl"), "paths.validation_jsonl"
    )
    if validation_path == train_path:
        raise ValueError("paths.train_jsonl and paths.validation_jsonl must be different files")
    output_dir = _configured_path(paths.get("adapter_dir"), "paths.adapter_dir", create_parent=True)
    if output_dir.exists() and any(output_dir.iterdir()) and args.resume_from_checkpoint is None:
        raise FileExistsError(
            f"adapter output is non-empty: {output_dir}; use --resume-from-checkpoint latest "
            "or choose a new output directory"
        )
    base_model = _configured_identifier(model_cfg.get("base_model"), "model.base_model")
    max_length = int(model_cfg.get("max_seq_length", 4096))
    local_only = bool(model_cfg.get("local_files_only", False))

    train_records = load_jsonl_records(
        train_path,
        require_answer=True,
        max_records=args.max_samples,
        expected_source_split="train",
        require_event_id=True,
    )
    validation_records = load_jsonl_records(
        validation_path,
        require_answer=True,
        expected_source_split="validation",
        require_event_id=True,
    )
    _validate_training_split_isolation(train_records, validation_records)
    tokenizer = load_tokenizer(base_model, local_files_only=local_only)
    train_dataset = TokenizedSFTDataset(train_records, tokenizer, max_length=max_length)
    eval_dataset = TokenizedSFTDataset(validation_records, tokenizer, max_length=max_length)

    quantized_4bit = bool(model_cfg.get("load_in_4bit", False))
    quantized_8bit = bool(model_cfg.get("load_in_8bit", False))
    load_settings = ModelLoadSettings(
        load_in_4bit=quantized_4bit,
        load_in_8bit=quantized_8bit,
        torch_dtype=str(model_cfg.get("torch_dtype", "auto")),
        device_map=model_cfg.get(
            "device_map", "auto" if (quantized_4bit or quantized_8bit) else None
        ),
        local_files_only=local_only,
    )
    model = load_causal_lm(base_model, load_settings)
    target_modules = lora_cfg.get(
        "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    if not isinstance(target_modules, list):
        raise ValueError("lora.target_modules must be a JSON array")
    lora_settings = LoraSettings(
        r=int(lora_cfg.get("r", 64)),
        alpha=int(lora_cfg.get("alpha", 16)),
        dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=tuple(str(item) for item in target_modules),
    )
    gradient_checkpointing = bool(training_cfg.get("gradient_checkpointing", True))
    model = attach_lora_for_training(
        model,
        lora_settings,
        quantized=quantized_4bit or quantized_8bit,
        gradient_checkpointing=gradient_checkpointing,
    )
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    training_kwargs = _training_argument_kwargs(
        training_cfg,
        output_dir=output_dir,
        seed=seed,
        gradient_checkpointing=gradient_checkpointing,
        training_arguments_class=TrainingArguments,
    )
    training_args = TrainingArguments(**training_kwargs)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": AnswerOnlyDataCollator(tokenizer),
    }
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    resume = args.resume_from_checkpoint
    if resume == "latest":
        resume = get_last_checkpoint(str(output_dir))
        if resume is None:
            raise FileNotFoundError(f"no Trainer checkpoint found below {output_dir}")
        resume = str(_safe_trainer_checkpoint(output_dir, resume))
    elif resume is not None:
        resume = str(_safe_trainer_checkpoint(output_dir, resume))
    trainer.train(resume_from_checkpoint=resume)
    best_checkpoint, best_validation_loss = _validated_best_selection(trainer, output_dir)
    _wait_for_everyone(trainer)
    if trainer.is_world_process_zero():
        # Export the selected checkpoint files themselves. This is fail-closed
        # even if a particular Trainer/PEFT version fails to restore the best
        # adapter into the in-memory model at the end of training.
        _export_best_adapter(best_checkpoint, output_dir)
        tokenizer.save_pretrained(str(output_dir), safe_serialization=True)
        _write_training_manifest(
            output_dir,
            base_model=base_model,
            train_count=len(train_dataset),
            validation_count=len(eval_dataset),
            best_checkpoint=best_checkpoint.name,
            best_validation_loss=best_validation_loss,
            seed=seed,
            max_length=max_length,
            lora=lora_settings,
            quantization="4bit" if quantized_4bit else "8bit" if quantized_8bit else "none",
        )
    _wait_for_everyone(trainer)
    return output_dir


def _load_json_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"LC-LLM configuration not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("LC-LLM configuration must be a JSON object")
    return payload


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = config.get(name)
    if not isinstance(section, Mapping):
        raise ValueError(f"missing configuration object: {name}")
    return section


def _configured_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or _placeholder(value):
        raise ValueError(f"{field} must be set in a private configuration")
    return value.strip()


def _configured_path(value: Any, field: str, *, create_parent: bool = False) -> Path:
    raw = _configured_identifier(value, field)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = BASELINE_ROOT / path
    path = path.resolve()
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _placeholder(value: str) -> bool:
    stripped = value.strip()
    return stripped.startswith("<") and stripped.endswith(">")


def _safe_trainer_checkpoint(output_dir: Path, value: str | Path) -> Path:
    checkpoint = Path(value).expanduser().resolve()
    root = output_dir.resolve()
    try:
        checkpoint.relative_to(root)
    except ValueError as exc:
        raise ValueError("resume checkpoint must be inside the configured adapter directory") from exc
    if not checkpoint.is_dir() or not re.fullmatch(r"checkpoint-\d+", checkpoint.name):
        raise ValueError(f"invalid local Trainer checkpoint directory: {checkpoint}")
    return checkpoint


def _training_argument_kwargs(
    training_cfg: Mapping[str, Any],
    *,
    output_dir: Path,
    seed: int,
    gradient_checkpointing: bool,
    training_arguments_class: Any,
) -> dict[str, Any]:
    """Build an auditable validation-selection schedule across HF versions."""

    save_steps = int(training_cfg.get("save_steps", 200))
    eval_steps = int(training_cfg.get("eval_steps", 200))
    if save_steps <= 0 or eval_steps <= 0:
        raise ValueError("training.save_steps and training.eval_steps must be positive")
    if save_steps != eval_steps:
        raise ValueError(
            "training.save_steps must equal training.eval_steps so every validation "
            "measurement corresponds to a selectable checkpoint"
        )
    save_total_limit = int(training_cfg.get("save_total_limit", 2))
    if save_total_limit <= 0:
        raise ValueError("training.save_total_limit must be positive")
    parameters = inspect.signature(training_arguments_class.__init__).parameters
    strategy_key = "eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"
    return {
        "output_dir": str(output_dir),
        "num_train_epochs": float(training_cfg.get("epochs", 2)),
        "per_device_train_batch_size": int(training_cfg.get("batch_size", 1)),
        "per_device_eval_batch_size": int(training_cfg.get("eval_batch_size", 1)),
        "gradient_accumulation_steps": int(
            training_cfg.get("gradient_accumulation_steps", 8)
        ),
        "learning_rate": float(training_cfg.get("learning_rate", 5e-4)),
        "weight_decay": float(training_cfg.get("weight_decay", 0.0)),
        "warmup_steps": int(training_cfg.get("warmup_steps", 600)),
        "logging_steps": int(training_cfg.get("logging_steps", 10)),
        "save_steps": save_steps,
        "eval_steps": eval_steps,
        "save_strategy": "steps",
        strategy_key: "steps",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_safetensors": True,
        "logging_strategy": "steps",
        "save_total_limit": save_total_limit,
        "seed": seed,
        "data_seed": seed,
        "fp16": bool(training_cfg.get("fp16", False)),
        "bf16": bool(training_cfg.get("bf16", False)),
        "gradient_checkpointing": gradient_checkpointing,
        "remove_unused_columns": True,
        "report_to": [],
        "optim": str(training_cfg.get("optim", "adamw_torch")),
        "dataloader_num_workers": int(training_cfg.get("num_workers", 0)),
        "ddp_find_unused_parameters": False,
    }


def _validate_training_split_isolation(
    train_records: list[Any], validation_records: list[Any]
) -> None:
    train_ids = {record.sample_id for record in train_records}
    validation_ids = {record.sample_id for record in validation_records}
    duplicate_ids = sorted(train_ids & validation_ids)
    if duplicate_ids:
        raise ValueError(
            "train and validation JSONL overlap in sample_id: "
            + ", ".join(duplicate_ids[:5])
        )
    train_events = {record.event_id for record in train_records}
    validation_events = {record.event_id for record in validation_records}
    duplicate_events = sorted(train_events & validation_events)
    if duplicate_events:
        raise ValueError(
            "train and validation JSONL overlap in event_id: "
            + ", ".join(str(item) for item in duplicate_events[:5])
        )


def _validated_best_selection(trainer: Any, output_dir: Path) -> tuple[Path, float]:
    """Require evidence that Trainer selected a materialized validation winner."""

    state = getattr(trainer, "state", None)
    raw_checkpoint = getattr(state, "best_model_checkpoint", None)
    raw_metric = getattr(state, "best_metric", None)
    if not raw_checkpoint or raw_metric is None:
        raise RuntimeError(
            "training completed without a selectable validation checkpoint; reduce "
            "training.eval_steps and training.save_steps so validation runs at least once"
        )
    metric = float(raw_metric)
    if not math.isfinite(metric):
        raise RuntimeError(f"best validation loss is not finite: {metric}")
    candidate = Path(raw_checkpoint).expanduser()
    if not candidate.is_absolute():
        candidate = output_dir / candidate
    checkpoint = _safe_trainer_checkpoint(output_dir, candidate)
    _required_adapter_artifacts(checkpoint)
    return checkpoint, metric


def _required_adapter_artifacts(checkpoint: Path) -> tuple[Path, Path]:
    weights = checkpoint / "adapter_model.safetensors"
    config = checkpoint / "adapter_config.json"
    missing = [path.name for path in (weights, config) if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"selected validation checkpoint {checkpoint.name} is missing LoRA artifacts: "
            + ", ".join(missing)
        )
    return weights, config


def _export_best_adapter(checkpoint: Path, output_dir: Path) -> None:
    """Copy the validation-selected LoRA adapter to the stable inference path."""

    weights, config = _required_adapter_artifacts(checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, output_dir / weights.name)
    shutil.copy2(config, output_dir / config.name)


def _wait_for_everyone(trainer: Any) -> None:
    accelerator = getattr(trainer, "accelerator", None)
    wait = getattr(accelerator, "wait_for_everyone", None)
    if callable(wait):
        wait()


def _write_training_manifest(
    output_dir: Path,
    *,
    base_model: str,
    train_count: int,
    validation_count: int,
    best_checkpoint: str,
    best_validation_loss: float,
    seed: int,
    max_length: int,
    lora: Any,
    quantization: str,
) -> None:
    # Do not record private dataset paths, prompts, answers, credentials, or host details.
    payload = {
        "baseline": "LC-LLM (adapted)",
        "base_model": base_model,
        "train_records": train_count,
        "validation_records": validation_count,
        "model_selection": {
            "split": "validation",
            "metric": "eval_loss",
            "greater_is_better": False,
            "best_checkpoint": best_checkpoint,
            "best_validation_loss": best_validation_loss,
            "test_records_used": 0,
        },
        "seed": seed,
        "max_seq_length": max_length,
        "quantization": quantization,
        "lora": {
            "r": lora.r,
            "alpha": lora.alpha,
            "dropout": lora.dropout,
            "target_modules": list(lora.target_modules),
        },
    }
    manifest = output_dir / "training_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    output = train(_arguments())
    print(output)


if __name__ == "__main__":
    main()
