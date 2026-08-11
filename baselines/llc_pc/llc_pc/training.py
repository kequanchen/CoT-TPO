"""Training and evaluation helpers for the data-free LLC-PC release."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .context_index import QueryResult, TrainContextIndex
from .intention_points import IntentionPointKMeans
from .model import LLCPCModelConfig, LLCPCMotionTransformer
from .tensorizer import (
    FeatureStandardizer,
    TensorizedBatch,
    standardizer_path_for_index,
)


def set_reproducible_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without changing global algorithms."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    requested = str(requested).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def retrieve_semantic_contexts(
    batch: TensorizedBatch,
    config: Mapping[str, Any],
    *,
    split: str,
) -> QueryResult:
    """Query a frozen training-only context index using observations only."""

    normalized_split = str(split).strip().lower()
    if normalized_split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    index_path = Path(config["paths"]["context_index"])
    index = TrainContextIndex.load(index_path)
    standardizer = FeatureStandardizer.load(standardizer_path_for_index(index_path))
    features = standardizer.transform(batch.retrieval_features)
    context_cfg = config["context"]
    exclude_event = bool(context_cfg.get("exclude_same_event", False))
    return index.query(
        features,
        k=int(context_cfg["k"]),
        sample_ids=batch.sample_ids if normalized_split == "train" else None,
        event_ids=batch.event_ids if exclude_event else None,
        exclude_self=normalized_split == "train",
        exclude_same_event=exclude_event,
    )


def load_intention_points(config: Mapping[str, Any]) -> np.ndarray:
    estimator = IntentionPointKMeans.load(config["paths"]["intention_points"])
    assert estimator.cluster_centers_ is not None
    expected = int(config["model"]["num_queries"])
    if estimator.cluster_centers_.shape != (expected, 2):
        raise ValueError(
            f"intention points have shape {estimator.cluster_centers_.shape}, "
            f"expected ({expected}, 2)"
        )
    return estimator.cluster_centers_.copy()


def model_config_from_public_config(config: Mapping[str, Any]) -> LLCPCModelConfig:
    model = config["model"]
    data = config["data"]
    d_model = int(model["d_model"])
    return LLCPCModelConfig(
        agent_feature_dim=7,
        map_feature_dim=2,
        context_dim=int(config["context"]["dimension"]),
        context_window=int(config["context"]["k"]),
        d_model=d_model,
        nhead=int(model["nhead"]),
        agent_encoder_layers=int(model["encoder_layers"]),
        decoder_layers=int(model["decoder_layers"]),
        dim_feedforward=d_model * 2,
        dropout=float(model["dropout"]),
        max_history_steps=int(data["history_steps"]),
        max_agents=7,
        future_steps=int(data["future_steps"]),
        num_output_modes=int(model["num_output_modes"]),
    )


def build_model(
    config: Mapping[str, Any], intention_points: Optional[np.ndarray] = None
) -> LLCPCMotionTransformer:
    points = load_intention_points(config) if intention_points is None else intention_points
    points = np.asarray(points, dtype=np.float32)
    expected = int(config["model"]["num_queries"])
    if points.shape != (expected, 2):
        raise ValueError(f"intention_points must have shape ({expected}, 2)")
    return LLCPCMotionTransformer(
        model_config_from_public_config(config), torch.from_numpy(points)
    )


class LLCPCArrayDataset(Dataset[dict[str, Tensor]]):
    """In-memory tensor dataset constructed from private data at runtime."""

    def __init__(
        self,
        batch: TensorizedBatch,
        retrieved: QueryResult,
        *,
        require_future: bool = True,
    ) -> None:
        if len(batch) != retrieved.contexts.shape[0]:
            raise ValueError("retrieved contexts must align with tensorized samples")
        if require_future and (batch.future is None or batch.future_valid_mask is None):
            raise ValueError("future labels are required for training/evaluation")
        self.batch = batch
        self.retrieved = retrieved
        self.require_future = require_future

    def __len__(self) -> int:
        return len(self.batch)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        item = {
            "agent_histories": torch.from_numpy(self.batch.agent_histories[index]),
            "agent_valid_mask": torch.from_numpy(self.batch.agent_valid_mask[index]),
            "map_polylines": torch.from_numpy(self.batch.map_polylines[index]),
            "map_valid_mask": torch.from_numpy(self.batch.map_valid_mask[index]),
            "semantic_contexts": torch.from_numpy(self.retrieved.contexts[index]),
            "semantic_context_mask": torch.from_numpy(self.retrieved.valid_mask[index]),
        }
        if self.batch.future is not None and self.batch.future_valid_mask is not None:
            item["future"] = torch.from_numpy(self.batch.future[index])
            item["future_valid_mask"] = torch.from_numpy(
                self.batch.future_valid_mask[index]
            )
        return item


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def move_batch(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def model_inputs(batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {
        "agent_histories": batch["agent_histories"].float(),
        "agent_valid_mask": batch["agent_valid_mask"].bool(),
        "map_polylines": batch["map_polylines"].float(),
        "map_valid_mask": batch["map_valid_mask"].bool(),
        "semantic_contexts": batch["semantic_contexts"].float(),
        "semantic_context_mask": batch["semantic_context_mask"].bool(),
    }


def save_checkpoint(
    path: Path,
    model: LLCPCMotionTransformer,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    config: Mapping[str, Any],
    metrics: Mapping[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "epoch": int(epoch),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": asdict(model.cfg),
            "metrics": dict(metrics),
            # Paths can be private, so only method hyperparameters are stored.
            "public_config": {
                section: dict(config[section])
                for section in (
                    "context",
                    "intention_points",
                    "model",
                    "train",
                    "evaluation",
                )
            },
        },
        path,
    )


def load_model_checkpoint(
    checkpoint: Path,
    config: Mapping[str, Any],
    device: torch.device,
) -> LLCPCMotionTransformer:
    # Restrict PyTorch to tensor and primitive checkpoint contents so an
    # untrusted pickle cannot execute arbitrary code during evaluation.
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "this PyTorch release does not support safe checkpoint loading; "
            "upgrade PyTorch instead of loading the checkpoint as unrestricted pickle"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("invalid LLC-PC checkpoint payload")
    if int(payload.get("format_version", -1)) != 1:
        raise ValueError("unsupported LLC-PC checkpoint format")
    expected_model_config = asdict(model_config_from_public_config(config))
    stored_model_config = payload.get("model_config")
    if stored_model_config != expected_model_config:
        raise ValueError(
            "checkpoint architecture does not match the current LLC-PC configuration"
        )
    if not isinstance(payload.get("model_state"), Mapping):
        raise ValueError("checkpoint does not contain a valid model state")
    model = build_model(config)
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
