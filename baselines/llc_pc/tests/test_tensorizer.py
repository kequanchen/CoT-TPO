from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.io


BASELINE_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from llc_pc import (  # noqa: E402
    FeatureStandardizer,
    TensorizerConfig,
    adapt_sample,
    dataset_layout_from_config,
    load_config,
    load_configured_split,
    standardizer_path_for_index,
    tensorize_sample,
    tensorize_samples,
)


def _raw_sample(future_offset: float = 0.0) -> dict:
    steps = 10
    x = np.arange(steps, dtype=np.float32)
    y = np.zeros(steps, dtype=np.float32)
    history = np.column_stack(
        (x, y, np.ones(steps), np.zeros(steps), np.zeros(steps), np.zeros(steps))
    ).astype(np.float32)
    ego = np.zeros((steps, 19), dtype=np.float32)
    ego[:, 0] = 10
    ego[:, 1] = 2
    ego[:, 2] = np.arange(steps)
    ego[:, 3] = x
    ego[:, 4] = y
    ego[:, 5] = 2.0
    ego[:, 6] = 2.0
    ego[:, 16] = 1.0
    leader = np.zeros((steps, 8), dtype=np.float32)
    leader[:, 0] = 11
    leader[:, 1] = 2
    leader[:, 2] = x + 10.0
    leader[:, 3] = 0.5
    leader[:, 4] = 1.0
    missing = np.full((steps, 8), np.nan, dtype=np.float32)
    future_x = x[-1] + np.arange(1, 51, dtype=np.float32) * 0.1
    future = np.column_stack((future_x + future_offset, np.zeros(50, dtype=np.float32)))
    return {
        "scenario_id": "event-a",
        "traj_id": 1,
        "lane_status": 1,
        "time_since_crossing": 0.0,
        "x_hist": history,
        "y_future": future,
        "ctx": {
            "ego": ego,
            "phys_tl": leader,
            "phys_tf": missing,
            "phys_tff": missing,
            "phys_ol": missing,
            "phys_of": missing,
            "phys_off": missing,
        },
    }


class TensorizerTests(unittest.TestCase):
    def test_fixed_shapes_masks_and_local_future(self) -> None:
        tensor = tensorize_sample(adapt_sample(_raw_sample()))
        self.assertEqual(tensor.agent_histories.shape, (7, 10, 7))
        self.assertEqual(tensor.agent_valid_mask.shape, (7, 10))
        self.assertEqual(tensor.map_polylines.shape, (4, 20, 2))
        self.assertEqual(tensor.map_valid_mask.shape, (4, 20))
        self.assertEqual(tensor.future.shape, (50, 2))
        self.assertEqual(tensor.future_valid_mask.shape, (50,))
        np.testing.assert_allclose(tensor.agent_histories[0, -1, :2], [0.0, 0.0])
        np.testing.assert_allclose(tensor.future[0], [0.1, 0.0], atol=1e-6)
        self.assertTrue(tensor.agent_valid_mask[0].all())
        self.assertTrue(tensor.agent_valid_mask[1].all())
        self.assertFalse(tensor.agent_valid_mask[2:].any())
        np.testing.assert_array_equal(tensor.agent_histories[2:], 0.0)

    def test_retrieval_feature_is_independent_of_future_labels(self) -> None:
        first = tensorize_sample(adapt_sample(_raw_sample(future_offset=0.0)))
        second = tensorize_sample(adapt_sample(_raw_sample(future_offset=5000.0)))
        np.testing.assert_array_equal(first.retrieval_features, second.retrieval_features)
        self.assertFalse(np.array_equal(first.future, second.future))

    def test_batch_and_training_only_standardizer_round_trip(self) -> None:
        raw_second = copy.deepcopy(_raw_sample())
        raw_second["traj_id"] = 2
        raw_second["x_hist"] = raw_second["x_hist"].copy()
        raw_second["ctx"]["ego"] = raw_second["ctx"]["ego"].copy()
        raw_second["x_hist"][:, 2] += 1.0
        batch = tensorize_samples([adapt_sample(_raw_sample()), adapt_sample(raw_second)])
        self.assertEqual(batch.agent_histories.shape[:3], (2, 7, 10))
        standardizer = FeatureStandardizer()
        with self.assertRaises(ValueError):
            standardizer.fit(batch.retrieval_features, source_split="test")
        transformed = standardizer.fit_transform(batch.retrieval_features)
        self.assertTrue(np.isfinite(transformed).all())
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "train_context_index_standardizer.npz"
            standardizer.save(artifact)
            restored = FeatureStandardizer.load(artifact)
            np.testing.assert_allclose(restored.transform(batch.retrieval_features), transformed)
        index = Path("somewhere/context_index.npz")
        self.assertEqual(
            standardizer_path_for_index(index).name,
            "context_index_standardizer.npz",
        )

    def test_public_config_loads_without_private_data(self) -> None:
        path = BASELINE_ROOT / "configs" / "post_crash_lc.example.json"
        config = load_config(path)
        self.assertEqual(config["data"]["train_mat"], "<PATH_TO_TRAIN_MAT>")
        self.assertEqual(
            config["data"]["validation_mat"], "<PATH_TO_VALIDATION_MAT>"
        )
        self.assertEqual(config["data"]["validation_key"], "validation_data")
        self.assertTrue(Path(config["paths"]["context_index"]).is_absolute())
        layout = dataset_layout_from_config(config)
        self.assertEqual(layout.history_steps, 10)
        self.assertEqual(layout.prediction_seconds, 5.0)
        self.assertEqual(TensorizerConfig.from_config(config).future_steps, 50)

    def test_config_requires_an_explicit_validation_split(self) -> None:
        example = BASELINE_ROOT / "configs" / "post_crash_lc.example.json"
        payload = json.loads(example.read_text(encoding="utf-8"))
        del payload["data"]["validation_mat"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing_validation.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "data.validation_mat"):
                load_config(path)

    def test_validation_split_uses_its_own_mat_and_key(self) -> None:
        example = BASELINE_ROOT / "configs" / "post_crash_lc.example.json"
        payload = json.loads(example.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation_mat = root / "validation.mat"
            scipy.io.savemat(validation_mat, {"validation_data": _raw_sample()})
            payload["data"]["validation_mat"] = str(validation_mat)
            config_path = root / "validation_config.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")

            config = load_config(config_path)
            samples = load_configured_split(
                config, "validation", include_future=True
            )
            self.assertEqual(len(samples), 1)
            self.assertIsNotNone(samples[0].future)

            payload["data"]["validation_key"] = "test_data"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            config = load_config(config_path)
            with self.assertRaisesRegex(KeyError, "test_data"):
                load_configured_split(config, "validation", include_future=True)


if __name__ == "__main__":
    unittest.main()
