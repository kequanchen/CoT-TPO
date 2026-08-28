#!/usr/bin/env python3
"""Train the self-contained, domain-adapted LLC-PC predictor."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch


BASELINE_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from llc_pc import (  # noqa: E402
    TensorizerConfig,
    llc_pc_loss,
    load_config,
    load_configured_split,
    tensorize_samples,
)
from llc_pc.training import (  # noqa: E402
    LLCPCArrayDataset,
    build_model,
    evaluate_loss_epoch,
    make_loader,
    model_inputs,
    move_batch,
    resolve_device,
    retrieve_semantic_contexts,
    save_checkpoint,
    set_reproducible_seed,
    write_json,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional debug-only training sample limit"
    )
    parser.add_argument(
        "--validation-limit",
        type=int,
        default=None,
        help="Optional debug-only validation sample limit",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs")
    parser.add_argument("--max-batches", type=int, default=None, help="Debug-only batches per epoch")
    parser.add_argument("--device", default=None, help="Override train.device")
    return parser.parse_args()


def _load_supervised_dataset(
    config: dict,
    split: str,
    limit: int | None,
) -> LLCPCArrayDataset:
    samples = load_configured_split(config, split, include_future=True)
    if limit is not None:
        if limit <= 0:
            raise ValueError(f"{split} sample limit must be positive")
        samples = samples[:limit]
    tensors = tensorize_samples(samples, TensorizerConfig.from_config(config))
    retrieved = retrieve_semantic_contexts(tensors, config, split=split)
    if not retrieved.valid_mask.any(axis=1).all():
        missing = int((~retrieved.valid_mask.any(axis=1)).sum())
        raise RuntimeError(
            f"{missing} {split} samples have no eligible training semantic context; "
            "expand the training context database or relax event exclusion"
        )
    return LLCPCArrayDataset(tensors, retrieved, require_future=True)


def _require_disjoint_event_ids(
    train_dataset: LLCPCArrayDataset,
    validation_dataset: LLCPCArrayDataset,
) -> None:
    """Fail if any crash episode contributes windows to both splits."""

    train_events = {str(value) for value in train_dataset.batch.event_ids}
    validation_events = {str(value) for value in validation_dataset.batch.event_ids}
    overlap = sorted(train_events & validation_events)
    if overlap:
        raise ValueError(
            "training and validation splits overlap in crash episode IDs: "
            + ", ".join(overlap[:5])
        )


def train(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    settings = config["train"]
    seed = int(settings.get("seed", 42))
    set_reproducible_seed(seed)
    device = resolve_device(args.device or settings.get("device", "auto"))

    train_dataset = _load_supervised_dataset(config, "train", args.limit)
    validation_dataset = _load_supervised_dataset(
        config, "validation", getattr(args, "validation_limit", None)
    )
    _require_disjoint_event_ids(train_dataset, validation_dataset)
    train_loader = make_loader(
        train_dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        num_workers=int(settings.get("num_workers", 0)),
        seed=seed,
    )
    validation_loader = make_loader(
        validation_dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=False,
        num_workers=int(settings.get("num_workers", 0)),
        seed=seed,
    )

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings.get("weight_decay", 1e-4)),
    )
    epochs = int(args.epochs or settings["epochs"])
    if epochs <= 0:
        raise ValueError("number of epochs must be positive")

    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    history: list[dict[str, float | bool | str]] = []
    best_validation_loss = math.inf
    best_epoch: int | None = None
    best_path = checkpoint_dir / "best.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"loss": 0.0, "regression_loss": 0.0, "classification_loss": 0.0}
        examples = 0
        for batch_index, batch in enumerate(train_loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(**model_inputs(batch))
            losses = llc_pc_loss(
                prediction,
                batch["future"].float(),
                batch["future_valid_mask"].bool(),
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            count = int(batch["future"].shape[0])
            examples += count
            for key in totals:
                totals[key] += float(losses[key].detach().cpu()) * count
        if examples == 0:
            raise RuntimeError("no training batches were processed")
        train_metrics = {key: value / examples for key, value in totals.items()}
        validation_metrics = evaluate_loss_epoch(model, validation_loader, device)
        if not math.isfinite(validation_metrics["loss"]):
            raise RuntimeError("validation loss is not finite")
        is_best = validation_metrics["loss"] < best_validation_loss
        if is_best:
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch
        metrics: dict[str, float | bool | str] = {
            "epoch": float(epoch),
            # Preserve the original metric keys for consumers of format-v1
            # checkpoints; selection is explicitly based on validation_loss.
            "loss": train_metrics["loss"],
            "regression_loss": train_metrics["regression_loss"],
            "classification_loss": train_metrics["classification_loss"],
            "train_loss": train_metrics["loss"],
            "train_regression_loss": train_metrics["regression_loss"],
            "train_classification_loss": train_metrics["classification_loss"],
            "validation_loss": validation_metrics["loss"],
            "validation_regression_loss": validation_metrics["regression_loss"],
            "validation_classification_loss": validation_metrics["classification_loss"],
            "best_validation_loss": best_validation_loss,
            "is_best": is_best,
            "selection_split": "validation",
            "selection_metric": "validation_loss",
        }
        history.append(metrics)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={metrics['train_loss']:.6f} "
            f"train_reg={metrics['train_regression_loss']:.6f} "
            f"train_cls={metrics['train_classification_loss']:.6f} "
            f"validation_loss={metrics['validation_loss']:.6f} "
            f"validation_reg={metrics['validation_regression_loss']:.6f} "
            f"validation_cls={metrics['validation_classification_loss']:.6f}"
        )
        save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            epoch=epoch,
            config=config,
            metrics=metrics,
        )
        if is_best:
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch=epoch,
                config=config,
                metrics=metrics,
            )
    write_json(
        checkpoint_dir / "training_history.json",
        {
            "selection_split": "validation",
            "selection_metric": "validation_loss",
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "epochs": history,
        },
    )
    return best_path


def main() -> None:
    path = train(_arguments())
    print(path)


if __name__ == "__main__":
    main()
