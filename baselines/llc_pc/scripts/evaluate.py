#!/usr/bin/env python3
"""Evaluate LLC-PC with the highest-probability trajectory (top-1 ADE/FDE)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


BASELINE_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from llc_pc import (  # noqa: E402
    TensorizerConfig,
    ade_fde,
    load_config,
    load_configured_split,
    tensorize_samples,
    top1_trajectory,
)
from llc_pc.training import (  # noqa: E402
    LLCPCArrayDataset,
    load_model_checkpoint,
    make_loader,
    model_inputs,
    move_batch,
    resolve_device,
    retrieve_semantic_contexts,
    set_reproducible_seed,
    write_json,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Optional debug-only sample limit")
    parser.add_argument("--device", default=None, help="Override train.device")
    parser.add_argument("--output", default=None, help="Optional metrics JSON path")
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    settings = config["train"]
    seed = int(settings.get("seed", 42))
    set_reproducible_seed(seed)
    device = resolve_device(args.device or settings.get("device", "auto"))

    samples = load_configured_split(config, "test", include_future=True)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        samples = samples[: args.limit]
    tensors = tensorize_samples(samples, TensorizerConfig.from_config(config))
    retrieved = retrieve_semantic_contexts(tensors, config, split="test")
    if not retrieved.valid_mask.any(axis=1).all():
        missing = int((~retrieved.valid_mask.any(axis=1)).sum())
        raise RuntimeError(
            f"{missing} test samples have no eligible training semantic context"
        )
    dataset = LLCPCArrayDataset(tensors, retrieved, require_future=True)
    loader = make_loader(
        dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=False,
        num_workers=int(settings.get("num_workers", 0)),
        seed=seed,
    )

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    model = load_model_checkpoint(checkpoint, config, device)
    model.eval()

    sample_rate = float(config["data"]["sample_rate_hz"])
    future_steps = int(config["data"]["future_steps"])
    horizons = [float(value) for value in config["evaluation"]["horizons_seconds"]]
    horizon_steps = {}
    for seconds in horizons:
        steps = int(round(seconds * sample_rate))
        if steps <= 0 or steps > future_steps:
            raise ValueError(
                f"evaluation horizon {seconds:g}s maps to invalid step {steps}"
            )
        horizon_steps[seconds] = steps
    totals = {seconds: {"ade": 0.0, "fde": 0.0, "samples": 0} for seconds in horizons}

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            prediction = model(**model_inputs(batch))
            top1 = top1_trajectory(prediction)
            target = batch["future"].float()
            valid = batch["future_valid_mask"].bool()
            count = int(target.shape[0])
            for seconds, steps in horizon_steps.items():
                horizon_valid = valid[:, :steps]
                if (~horizon_valid.any(dim=-1)).any():
                    raise ValueError(f"a sample has no valid future point at {seconds:g}s")
                ade, fde = ade_fde(
                    top1[:, :steps], target[:, :steps], horizon_valid
                )
                totals[seconds]["ade"] += float(ade.cpu()) * count
                totals[seconds]["fde"] += float(fde.cpu()) * count
                totals[seconds]["samples"] += count

    metrics = {
        "prediction_mode": "top1",
        "checkpoint": checkpoint.name,
        "horizons": {
            f"{seconds:g}s": {
                "ADE": values["ade"] / values["samples"],
                "FDE": values["fde"] / values["samples"],
                "samples": values["samples"],
            }
            for seconds, values in totals.items()
        },
    }
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else Path(config["paths"]["work_dir"]) / "evaluation_top1.json"
    )
    write_json(output, metrics)
    for horizon, values in metrics["horizons"].items():
        print(f"{horizon}: ADE={values['ADE']:.6f} FDE={values['FDE']:.6f}")
    return output


def main() -> None:
    path = evaluate(_arguments())
    print(path)


if __name__ == "__main__":
    main()
