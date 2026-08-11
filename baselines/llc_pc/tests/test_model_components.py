"""Synthetic-data tests for leakage guards and motion-query conditioning."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


BASELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINE_ROOT))

from llc_pc.conditioning import ContextProjector, MotionQueryConditioner  # noqa: E402
from llc_pc.context_index import TrainContextIndex  # noqa: E402
from llc_pc.intention_points import (  # noqa: E402
    IntentionPointKMeans,
    extract_future_endpoints,
)


class TrainContextIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.embeddings = np.asarray(
            [[0.0, 0.0], [0.1, 0.0], [5.0, 0.0], [5.1, 0.0]], dtype=np.float32
        )
        self.contexts = np.arange(4 * 17, dtype=np.float32).reshape(4, 17)
        self.sample_ids = ["s0", "s1", "s2", "s3"]
        self.event_ids = ["event-a", "event-a", "event-b", "event-c"]

    def test_non_training_fit_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TrainContextIndex().fit(
                self.embeddings,
                self.contexts,
                self.sample_ids,
                source_split="test",
            )

    def test_self_and_event_exclusion(self) -> None:
        index = TrainContextIndex().fit(
            self.embeddings,
            self.contexts,
            self.sample_ids,
            event_ids=self.event_ids,
        )
        result = index.query(
            self.embeddings[:1],
            k=2,
            sample_ids=self.sample_ids[:1],
            event_ids=self.event_ids[:1],
            exclude_self=True,
            exclude_same_event=True,
        )
        np.testing.assert_array_equal(result.indices, [[2, 3]])
        np.testing.assert_array_equal(result.sample_ids, [["s2", "s3"]])
        self.assertTrue(result.valid_mask.all())
        np.testing.assert_array_equal(result.contexts[0, 0], self.contexts[2])

    def test_padding_and_round_trip(self) -> None:
        index = TrainContextIndex(metric="cosine").fit(
            self.embeddings + 1.0,
            self.contexts,
            self.sample_ids,
            event_ids=self.event_ids,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_context_index.npz"
            index.save(path)
            restored = TrainContextIndex.load(path)
            result = restored.query(
                np.asarray([[1.0, 1.0]], dtype=np.float32),
                k=5,
            )
        self.assertEqual(result.contexts.shape, (1, 5, 17))
        self.assertEqual(int(result.valid_mask.sum()), 4)
        self.assertEqual(int(result.indices[0, -1]), -1)
        self.assertTrue(np.isinf(result.distances[0, -1]))


class IntentionPointTests(unittest.TestCase):
    def test_endpoint_mask_uses_last_valid_step(self) -> None:
        trajectories = np.asarray(
            [
                [[0.0, 0.0], [1.0, 1.0], [99.0, 99.0]],
                [[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
            ],
            dtype=np.float32,
        )
        mask = np.asarray([[True, True, False], [True, True, True]])
        endpoints, indices = extract_future_endpoints(trajectories, mask)
        np.testing.assert_array_equal(endpoints, [[1.0, 1.0], [4.0, 4.0]])
        np.testing.assert_array_equal(indices, [0, 1])

    def test_two_cluster_fit_is_deterministic_and_serializable(self) -> None:
        rng = np.random.default_rng(7)
        endpoints = np.concatenate(
            [
                rng.normal([-2.0, 5.0], 0.05, size=(40, 2)),
                rng.normal([2.0, 10.0], 0.05, size=(40, 2)),
            ]
        ).astype(np.float32)
        futures = np.stack([np.zeros_like(endpoints), endpoints], axis=1)
        model = IntentionPointKMeans(n_clusters=2, random_state=11, n_init=4).fit(futures)
        centers = model.cluster_centers_[np.argsort(model.cluster_centers_[:, 0])]
        np.testing.assert_allclose(centers, [[-2.0, 5.0], [2.0, 10.0]], atol=0.08)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "intention_points.npz"
            model.save(path)
            restored = IntentionPointKMeans.load(path)
        np.testing.assert_allclose(restored.cluster_centers_, model.cluster_centers_)

    def test_non_training_fit_is_rejected(self) -> None:
        futures = np.asarray([[[0.0, 0.0]], [[1.0, 1.0]]], dtype=np.float32)
        with self.assertRaises(ValueError):
            IntentionPointKMeans(n_clusters=2).fit(futures, source_split="validation")


class MotionQueryConditioningTests(unittest.TestCase):
    def test_cyclic_assignment_and_shapes(self) -> None:
        torch.manual_seed(3)
        projector = ContextProjector(context_dim=17, hidden_dim=8, query_dim=6)
        conditioner = MotionQueryConditioner(projector)
        contexts = torch.randn(2, 3, 17, requires_grad=True)
        mask = torch.tensor([[True, False, True], [False, False, False]])
        position = torch.randn(5, 6)
        prepared = conditioner.prepare(contexts, position, context_mask=mask)
        self.assertEqual(tuple(prepared.query_content.shape), (2, 5, 6))
        self.assertEqual(tuple(prepared.query_position.shape), (2, 5, 6))
        self.assertEqual(prepared.context_assignment[0].tolist(), [0, 2, 0, 2, 0])
        self.assertEqual(prepared.context_assignment[1].tolist(), [-1, -1, -1, -1, -1])
        torch.testing.assert_close(prepared.query_content[1], torch.zeros(5, 6))
        prepared.query_content.sum().backward()
        self.assertIsNotNone(contexts.grad)

    def test_sequence_first_adapter_and_add_mode(self) -> None:
        projector = ContextProjector(context_dim=17, hidden_dim=8, query_dim=4)
        conditioner = MotionQueryConditioner(projector, combine="add")
        contexts = torch.zeros(2, 17)
        positions = torch.zeros(3, 4)
        base = torch.ones(3, 4)
        prepared = conditioner.prepare(
            contexts,
            positions,
            base_query_content=base,
            sequence_first=True,
        )
        self.assertEqual(tuple(prepared.query_content.shape), (3, 2, 4))
        self.assertEqual(tuple(prepared.query_position.shape), (3, 2, 4))
        self.assertFalse(prepared.batch_first)


if __name__ == "__main__":
    unittest.main()
