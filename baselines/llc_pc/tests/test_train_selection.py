"""Regression test for validation-only checkpoint selection in train.py."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


BASELINE_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))


def _load_train_script():
    path = BASELINE_ROOT / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("llc_pc_train_script", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self):
        return self.weight


class TrainSelectionTest(unittest.TestCase):
    def test_best_checkpoint_uses_validation_and_train_never_loads_test(self) -> None:
        train_script = _load_train_script()
        validation_losses = [3.0, 1.0, 2.0]
        batch = {
            "future": torch.zeros((1, 1, 2), dtype=torch.float32),
            "future_valid_mask": torch.ones((1, 1), dtype=torch.bool),
        }

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "checkpoints"
            config = {
                "train": {
                    "batch_size": 1,
                    "epochs": 3,
                    "learning_rate": 0.1,
                    "weight_decay": 0.0,
                    "seed": 7,
                    "num_workers": 0,
                    "device": "cpu",
                },
                "paths": {"checkpoint_dir": str(checkpoint_dir)},
            }
            args = argparse.Namespace(
                config="ignored.json",
                limit=None,
                validation_limit=None,
                epochs=None,
                max_batches=None,
                device="cpu",
            )
            loaded_splits: list[str] = []
            loader_calls: list[tuple[str, bool]] = []

            def fake_load_dataset(_config, split, limit):
                self.assertIsNone(limit)
                loaded_splits.append(split)
                if split == "test":
                    self.fail("train.py must not load the test split")
                return SimpleNamespace(
                    name=split,
                    batch=SimpleNamespace(
                        event_ids=np.asarray([f"{split}-episode"], dtype=str)
                    ),
                )

            def fake_make_loader(dataset, *, shuffle, **_kwargs):
                loader_calls.append((dataset.name, shuffle))
                return [batch] if dataset.name == "train" else ["validation-batch"]

            def fake_loss(prediction, _future, _valid):
                loss = prediction.square()
                return {
                    "loss": loss,
                    "regression_loss": loss * 0.75,
                    "classification_loss": loss * 0.25,
                }

            def fake_save(path, _model, _optimizer, *, epoch, config, metrics):
                del config
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"epoch": epoch, "metrics": dict(metrics)}, path)

            validation_metrics = [
                {
                    "loss": value,
                    "regression_loss": value * 0.75,
                    "classification_loss": value * 0.25,
                }
                for value in validation_losses
            ]

            with (
                mock.patch.object(train_script, "load_config", return_value=config),
                mock.patch.object(
                    train_script,
                    "_load_supervised_dataset",
                    side_effect=fake_load_dataset,
                ),
                mock.patch.object(
                    train_script, "make_loader", side_effect=fake_make_loader
                ),
                mock.patch.object(train_script, "build_model", return_value=_TinyModel()),
                mock.patch.object(train_script, "move_batch", side_effect=lambda value, _device: value),
                mock.patch.object(train_script, "model_inputs", return_value={}),
                mock.patch.object(train_script, "llc_pc_loss", side_effect=fake_loss),
                mock.patch.object(
                    train_script,
                    "evaluate_loss_epoch",
                    side_effect=validation_metrics,
                ),
                mock.patch.object(train_script, "save_checkpoint", side_effect=fake_save),
            ):
                best_path = train_script.train(args)

            self.assertEqual(loaded_splits, ["train", "validation"])
            self.assertEqual(loader_calls, [("train", True), ("validation", False)])
            self.assertEqual(best_path, checkpoint_dir / "best.pt")
            best = torch.load(best_path, map_location="cpu", weights_only=True)
            last = torch.load(
                checkpoint_dir / "last.pt", map_location="cpu", weights_only=True
            )
            self.assertEqual(best["epoch"], 2)
            self.assertEqual(last["epoch"], 3)
            self.assertEqual(best["metrics"]["selection_split"], "validation")
            self.assertEqual(
                best["metrics"]["selection_metric"], "validation_loss"
            )

            history = json.loads(
                (checkpoint_dir / "training_history.json").read_text(encoding="utf-8")
            )
            self.assertEqual(history["selection_split"], "validation")
            self.assertEqual(history["selection_metric"], "validation_loss")
            self.assertEqual(history["best_epoch"], 2)
            self.assertEqual(history["best_validation_loss"], 1.0)

    def test_crash_episode_overlap_is_rejected(self) -> None:
        train_script = _load_train_script()
        train_dataset = SimpleNamespace(
            batch=SimpleNamespace(event_ids=np.asarray(["crash-1", "crash-2"]))
        )
        validation_dataset = SimpleNamespace(
            batch=SimpleNamespace(event_ids=np.asarray(["crash-2", "crash-3"]))
        )
        with self.assertRaisesRegex(ValueError, "crash episode IDs"):
            train_script._require_disjoint_event_ids(
                train_dataset, validation_dataset
            )


if __name__ == "__main__":
    unittest.main()
