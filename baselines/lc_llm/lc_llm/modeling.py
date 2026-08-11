"""Lazy, security-conscious model and PEFT loading for adapted LC-LLM."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class LoraSettings:
    """LoRA hyperparameters reported by LC-LLM, with explicit target modules."""

    r: int = 64
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    def validate(self) -> None:
        if self.r <= 0 or self.alpha <= 0:
            raise ValueError("LoRA r and alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if not self.target_modules or any(not item.strip() for item in self.target_modules):
            raise ValueError("at least one non-empty LoRA target module is required")


@dataclass(frozen=True)
class ModelLoadSettings:
    """Base-model loading options shared by training and prediction."""

    load_in_4bit: bool = False
    load_in_8bit: bool = False
    torch_dtype: str = "auto"
    device_map: str | Mapping[str, Any] | None = "auto"
    local_files_only: bool = False

    def validate(self) -> None:
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("4-bit and 8-bit loading are mutually exclusive")
        if self.torch_dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError(
                "torch_dtype must be one of: auto, float32, float16, bfloat16"
            )


def set_global_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy (when installed), and PyTorch (when installed)."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def load_tokenizer(model_name_or_path: str, *, local_files_only: bool = False) -> Any:
    """Load a tokenizer without permitting repository-defined Python code."""

    _require_identifier(model_name_or_path, "model_name_or_path")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install 'transformers' to load the LC-LLM tokenizer") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=False,
        trust_remote_code=False,
        local_files_only=local_files_only,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("the base tokenizer must define an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_causal_lm(
    model_name_or_path: str,
    settings: ModelLoadSettings,
) -> Any:
    """Load a causal LM with optional bitsandbytes quantization.

    ``trust_remote_code`` is deliberately fixed to ``False`` and safe tensor
    weights are required.  Quantization dependencies are imported by
    Transformers only when a quantized mode is requested.
    """

    _require_identifier(model_name_or_path, "model_name_or_path")
    settings.validate()
    try:
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("install 'torch' and 'transformers' to load LC-LLM") from exc
    kwargs: dict[str, Any] = {
        "trust_remote_code": False,
        "local_files_only": settings.local_files_only,
        "use_safetensors": True,
    }
    if settings.device_map is not None:
        kwargs["device_map"] = settings.device_map
    dtype = _torch_dtype(torch, settings.torch_dtype)
    if dtype != "auto":
        kwargs["torch_dtype"] = dtype
    if settings.load_in_4bit or settings.load_in_8bit:
        compute_dtype = dtype if dtype != "auto" else torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=settings.load_in_4bit,
            load_in_8bit=settings.load_in_8bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)


def attach_lora_for_training(
    model: Any,
    settings: LoraSettings,
    *,
    quantized: bool,
    gradient_checkpointing: bool = False,
) -> Any:
    """Attach trainable PEFT LoRA modules to a loaded causal LM."""

    settings.validate()
    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise RuntimeError("install 'peft' to train the LC-LLM LoRA adapter") from exc
    if quantized:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
        )
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if hasattr(model, "config"):
        model.config.use_cache = False
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=settings.r,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        target_modules=list(settings.target_modules),
        bias="none",
    )
    return get_peft_model(model, config)


def validate_adapter_directory(
    adapter_dir: str | Path,
    *,
    expected_base_model: str | None = None,
) -> dict[str, Any]:
    """Validate a local, safetensors-only LoRA adapter before loading it.

    Requiring a local directory avoids silently executing or replacing adapter
    artifacts from a mutable Hub repository at evaluation time.  The function
    also verifies the adapter type, task type, and optional base-model identity.
    """

    root = Path(adapter_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"local LC-LLM adapter directory not found: {root}")
    unsafe_weights = sorted(
        item.name for item in root.iterdir() if item.is_file() and _is_unsafe_weight_name(item.name)
    )
    if unsafe_weights:
        raise ValueError(
            "adapter directory contains pickle-compatible weight artifacts; retain only "
            f"safetensors weights: {', '.join(unsafe_weights)}"
        )
    config_path = _contained_file(root, "adapter_config.json")
    weights_path = _contained_file(root, "adapter_model.safetensors")
    if weights_path.suffix != ".safetensors":  # defensive, despite fixed filename
        raise ValueError("adapter weights must use safetensors")
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid adapter configuration: {config_path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError("adapter_config.json must contain a JSON object")
    if str(metadata.get("peft_type", "")).upper() != "LORA":
        raise ValueError("only a PEFT LoRA adapter is accepted")
    task_type = str(metadata.get("task_type", "")).upper()
    if task_type not in {"CAUSAL_LM", "CAUSAL_LM_TASK"}:
        raise ValueError(f"adapter task_type must be CAUSAL_LM, got {task_type!r}")
    base_model = _require_identifier(
        metadata.get("base_model_name_or_path"),
        "adapter base_model_name_or_path",
    )
    if expected_base_model is not None and not _same_model_identifier(
        base_model, expected_base_model
    ):
        raise ValueError(
            "adapter base model does not match --base-model: "
            f"{base_model!r} != {expected_base_model!r}"
        )
    metadata = dict(metadata)
    metadata["_validated_adapter_dir"] = str(root)
    return metadata


def load_adapter_for_inference(
    adapter_dir: str | Path,
    *,
    expected_base_model: str | None = None,
    settings: ModelLoadSettings | None = None,
) -> tuple[Any, str]:
    """Safely load a local LoRA adapter and return ``(model, base_model_id)``."""

    metadata = validate_adapter_directory(
        adapter_dir,
        expected_base_model=expected_base_model,
    )
    base_model = str(metadata["base_model_name_or_path"])
    model = load_causal_lm(base_model, settings or ModelLoadSettings())
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("install 'peft' to load the LC-LLM adapter") from exc
    adapter_path = str(metadata["_validated_adapter_dir"])
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    return model, base_model


def _torch_dtype(torch_module: Any, name: str) -> Any:
    return {
        "auto": "auto",
        "float32": torch_module.float32,
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
    }[name]


def _contained_file(root: Path, name: str) -> Path:
    candidate = (root / name).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"adapter artifact escapes its directory: {name}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"required adapter artifact not found: {candidate}")
    return candidate


def _is_unsafe_weight_name(name: str) -> bool:
    """Identify pickle weight names that model loaders may auto-discover.

    ``Trainer`` legitimately writes metadata such as ``training_args.bin``;
    that file is not read by PEFT inference and is therefore allowed.  The
    adapter/base weight filenames below are loadable alternatives to the
    required safetensors artifact and must not coexist with it.
    """

    lowered = name.casefold()
    if lowered in {
        "adapter_model.bin",
        "adapter_model.pt",
        "adapter_model.pth",
        "pytorch_model.bin",
        "pytorch_model.pt",
        "pytorch_model.pth",
        "model.bin",
        "model.pt",
        "model.pth",
    }:
        return True
    return lowered.startswith("pytorch_model-") and lowered.endswith((".bin", ".pt", ".pth"))


def _require_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _same_model_identifier(first: str, second: str) -> bool:
    if first.strip().rstrip("/") == second.strip().rstrip("/"):
        return True
    first_path = Path(first).expanduser()
    second_path = Path(second).expanduser()
    if first_path.exists() and second_path.exists():
        return first_path.resolve() == second_path.resolve()
    return False
