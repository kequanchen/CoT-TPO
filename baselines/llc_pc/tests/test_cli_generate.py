from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.io


BASELINE_ROOT = Path(__file__).resolve().parents[1]


def _mat_sample() -> dict:
    steps = 10
    x = np.arange(steps, dtype=np.float32)
    history = np.column_stack(
        (x, np.zeros(steps), np.ones(steps), np.zeros((steps, 3)))
    ).astype(np.float32)
    ego = np.zeros((steps, 19), dtype=np.float32)
    ego[:, 0] = 1
    ego[:, 2] = np.arange(steps)
    ego[:, 3] = x
    ego[:, 5:7] = 2.0
    missing = np.full((steps, 8), np.nan, dtype=np.float32)
    return {
        "scenario_id": "synthetic-event",
        "traj_id": 1,
        "lane_status": 1,
        "time_since_crossing": 0.0,
        "x_hist": history,
        "y_future": np.full((50, 2), 7777.0, dtype=np.float32),
        "ctx": {
            "ego": ego,
            "phys_tl": missing,
            "phys_tf": missing,
            "phys_tff": missing,
            "phys_ol": missing,
            "phys_of": missing,
            "phys_off": missing,
        },
    }


class GenerateContextsCliTests(unittest.TestCase):
    def test_dry_run_writes_only_observation_derived_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mat_path = root / "private.mat"
            scipy.io.savemat(mat_path, {"train_data": _mat_sample()})
            with (BASELINE_ROOT / "configs" / "post_crash_lc.example.json").open(
                "r", encoding="utf-8"
            ) as handle:
                config = json.load(handle)
            config["data"]["train_mat"] = str(mat_path)
            config["data"]["test_mat"] = str(mat_path)
            config["paths"] = {
                key: str(root / key) for key in config["paths"]
            }
            config_path = root / "private.local.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            script = BASELINE_ROOT / "scripts" / "generate_contexts.py"
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--config",
                    str(config_path),
                    "--split",
                    "train",
                    "--skip-llm",
                    "--max-samples",
                    "1",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            manifest = root / "raw_context_dir" / "train" / "contexts.jsonl"
            record = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "dry_run")
            prompt = (manifest.parent / record["prompt_file"]).read_text(encoding="utf-8")
            self.assertNotIn("7777", prompt)
            self.assertTrue((manifest.parent / record["tc_map_file"]).is_file())


if __name__ == "__main__":
    unittest.main()
