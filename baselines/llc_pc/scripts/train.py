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
    parser.add_argument("--limit", type=int, default=None, help="Optional debug-only sample limit")
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs")
    parser.add_argument("--max-batches", type=int, default=None, help="Debug-only batches per epoch")
    parser.add_argument("--device", default=None, help="Override train.device")
    return parser.parse_args()


def train(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    settings = config["train"]
    seed = int(settings.get("seed", 42))
    set_reproducible_seed(seed)
    device = resolve_device(args.device or settings.get("device", "auto"))

    samples = load_configured_split(config, "train", include_future=True)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        samples = samples[: args.limit]
    tensors = tensorize_samples(samples, TensorizerConfig.from_config(config))
    retrieved = retrieve_semantic_contexts(tensors, config, split="train")
    if not retrieved.valid_mask.any(axis=1).all():
        missing = int((~retrieved.valid_mask.any(axis=1)).sum())
        raise RuntimeError(
            f"{missing} training samples have no eligible semantic context; "
            "expand the training context database or relax event exclusion"
        )
    dataset = LLCPCArrayDataset(tensors, retrieved, require_future=True)
    loader = make_loader(
        dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=True,
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
    history: list[dict[str, float]] = []
    best_loss = math.inf
    best_path = checkpoint_dir / "best.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"loss": 0.0, "regression_loss": 0.0, "classification_loss": 0.0}
        examples = 0
        for batch_index, batch in enumerate(loader):
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
        metrics = {key: value / examples for key, value in totals.items()}
        metrics["epoch"] = float(epoch)
        history.append(metrics)
        print(
            f"epoch={epoch:03d} loss={metrics['loss']:.6f} "
            f"reg={metrics['regression_loss']:.6f} "
            f"cls={metrics['classification_loss']:.6f}"
        )
        save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            epoch=epoch,
            config=config,
            metrics=metrics,
        )
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch=epoch,
                config=config,
                metrics=metrics,
            )
    write_json(checkpoint_dir / "training_history.json", {"epochs": history})
    return best_path


def main() -> None:
    path = train(_arguments())
    print(path)


if __name__ == "__main__":
    main()
