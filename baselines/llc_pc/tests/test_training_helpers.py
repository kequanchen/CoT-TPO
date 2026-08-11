"""End-to-end synthetic test for LLC-PC artifact/model integration."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


BASELINE_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from llc_pc import (  # noqa: E402
    FeatureStandardizer,
    IntentionPointKMeans,
    TrainContextIndex,
    llc_pc_loss,
)
from llc_pc.tensorizer import TensorizedBatch, standardizer_path_for_index  # noqa: E402
from llc_pc.training import (  # noqa: E402
    LLCPCArrayDataset,
    build_model,
    load_model_checkpoint,
    model_inputs,
    retrieve_semantic_contexts,
    save_checkpoint,
)


class TrainingHelpersTest(unittest.TestCase):
    def test_artifacts_one_step_and_checkpoint(self) -> None:
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_path = root / "context_index.npz"
            intention_path = root / "intentions.npz"

            database_features = rng.normal(size=(8, 6)).astype(np.float32)
            standardizer = FeatureStandardizer()
            database_standard = standardizer.fit_transform(
                database_features, source_split="train"
            )
            database_contexts = rng.normal(size=(8, 17)).astype(np.float32)
            TrainContextIndex(context_dim=17).fit(
                database_standard,
                database_contexts,
                [f"db-{index}" for index in range(8)],
                event_ids=[f"event-{index}" for index in range(8)],
                source_split="train",
            ).save(index_path)
            standardizer.save(standardizer_path_for_index(index_path))

            cluster_futures = np.zeros((8, 5, 2), dtype=np.float32)
            cluster_futures[:, -1] = np.asarray(
                [[2, -1], [2, 1], [4, -1], [4, 1], [6, -2], [6, 2], [8, -2], [8, 2]],
                dtype=np.float32,
            )
            IntentionPointKMeans(
                n_clusters=4, random_state=3, n_init=2, max_iter=20
            ).fit(cluster_futures, source_split="train").save(intention_path)

            count, steps = 2, 5
            batch = TensorizedBatch(
                sample_ids=np.asarray(["query-0", "query-1"]),
                event_ids=np.asarray(["query-event-0", "query-event-1"]),
                agent_histories=rng.normal(size=(count, 7, 4, 7)).astype(np.float32),
                agent_valid_mask=np.ones((count, 7, 4), dtype=bool),
                map_polylines=rng.normal(size=(count, 4, 3, 2)).astype(np.float32),
                map_valid_mask=np.ones((count, 4, 3), dtype=bool),
                retrieval_features=rng.normal(size=(count, 6)).astype(np.float32),
                future=rng.normal(size=(count, steps, 2)).astype(np.float32),
                future_valid_mask=np.ones((count, steps), dtype=bool),
            )
            config = {
                "data": {"history_steps": 4, "future_steps": 5},
                "paths": {
                    "context_index": str(index_path),
                    "intention_points": str(intention_path),
                },
                "context": {
                    "dimension": 17,
                    "k": 2,
                    "exclude_same_event": False,
                },
                "intention_points": {"n_clusters": 4},
                "model": {
                    "d_model": 16,
                    "nhead": 4,
                    "encoder_layers": 1,
                    "decoder_layers": 1,
                    "dropout": 0.0,
                    "num_queries": 4,
                    "num_output_modes": 2,
                },
                "train": {"seed": 3},
                "evaluation": {"prediction_mode": "top1"},
            }
            retrieved = retrieve_semantic_contexts(batch, config, split="test")
            dataset = LLCPCArrayDataset(batch, retrieved)
            item = dataset[0]
            torch_batch = {key: value.unsqueeze(0) for key, value in item.items()}

            model = build_model(config)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            prediction = model(**model_inputs(torch_batch))
            losses = llc_pc_loss(
                prediction,
                torch_batch["future"].float(),
                torch_batch["future_valid_mask"].bool(),
            )
            losses["loss"].backward()
            optimizer.step()
            checkpoint = root / "model.pt"
            save_checkpoint(
                checkpoint,
                model,
                optimizer,
                epoch=1,
                config=config,
                metrics={"loss": float(losses["loss"].detach())},
            )
            loaded = load_model_checkpoint(checkpoint, config, torch.device("cpu"))
            self.assertEqual(loaded.intention_points.shape, (4, 2))
            mismatched = copy.deepcopy(config)
            mismatched["model"]["nhead"] = 2
            with self.assertRaisesRegex(ValueError, "architecture does not match"):
                load_model_checkpoint(checkpoint, mismatched, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
