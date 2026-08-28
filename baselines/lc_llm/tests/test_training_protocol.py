from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


BASELINE_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from lc_llm.config import load_config  # noqa: E402
from lc_llm.sft import SFTRecord  # noqa: E402


def load_script(name: str):
    path = BASELINE_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"lc_llm_test_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LegacyTrainingArguments:
    def __init__(self, *, evaluation_strategy=None):
        self.evaluation_strategy = evaluation_strategy


class ModernTrainingArguments:
    def __init__(self, *, eval_strategy=None):
        self.eval_strategy = eval_strategy


class TrainingProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train_script = load_script("train_lora")
        cls.prepare_script = load_script("prepare_dataset")

    def test_training_arguments_select_minimum_validation_loss(self) -> None:
        for arguments_class, strategy_key in (
            (LegacyTrainingArguments, "evaluation_strategy"),
            (ModernTrainingArguments, "eval_strategy"),
        ):
            kwargs = self.train_script._training_argument_kwargs(
                {"save_steps": 25, "eval_steps": 25},
                output_dir=Path("adapter"),
                seed=42,
                gradient_checkpointing=True,
                training_arguments_class=arguments_class,
            )
            self.assertEqual(kwargs[strategy_key], "steps")
            self.assertEqual(kwargs["save_strategy"], "steps")
            self.assertTrue(kwargs["load_best_model_at_end"])
            self.assertEqual(kwargs["metric_for_best_model"], "eval_loss")
            self.assertFalse(kwargs["greater_is_better"])

    def test_every_validation_measurement_must_be_a_checkpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "must equal"):
            self.train_script._training_argument_kwargs(
                {"save_steps": 50, "eval_steps": 25},
                output_dir=Path("adapter"),
                seed=42,
                gradient_checkpointing=False,
                training_arguments_class=LegacyTrainingArguments,
            )

    def test_train_validation_crash_episodes_must_be_disjoint(self) -> None:
        train = [SFTRecord("train-1", "s", "u", "a", "crash-1", "train")]
        validation = [
            SFTRecord("validation-1", "s", "u", "a", "crash-1", "validation")
        ]
        with self.assertRaisesRegex(ValueError, "overlap in event_id"):
            self.train_script._validate_training_split_isolation(train, validation)

    def test_best_validation_checkpoint_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            checkpoint = output_dir / "checkpoint-25"
            checkpoint.mkdir()
            (checkpoint / "adapter_model.safetensors").write_bytes(b"best-weights")
            (checkpoint / "adapter_config.json").write_text(
                '{"base_model_name_or_path": "base"}', encoding="utf-8"
            )

            class State:
                best_model_checkpoint = str(checkpoint)
                best_metric = 1.25

            class Trainer:
                state = State()

            selected, loss = self.train_script._validated_best_selection(
                Trainer(), output_dir
            )
            self.assertEqual(selected, checkpoint.resolve())
            self.assertEqual(loss, 1.25)

            (checkpoint / "adapter_model.safetensors").unlink()
            with self.assertRaisesRegex(RuntimeError, "missing LoRA artifacts"):
                self.train_script._validated_best_selection(Trainer(), output_dir)

            State.best_model_checkpoint = None
            with self.assertRaisesRegex(RuntimeError, "without a selectable validation"):
                self.train_script._validated_best_selection(Trainer(), output_dir)

    def test_stable_adapter_is_copied_from_selected_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "adapter"
            checkpoint = output_dir / "checkpoint-50"
            checkpoint.mkdir(parents=True)
            (checkpoint / "adapter_model.safetensors").write_bytes(b"selected")
            (checkpoint / "adapter_config.json").write_text(
                '{"selected": true}', encoding="utf-8"
            )
            (output_dir / "adapter_model.safetensors").write_bytes(b"stale-last")

            self.train_script._export_best_adapter(checkpoint, output_dir)

            self.assertEqual(
                (output_dir / "adapter_model.safetensors").read_bytes(), b"selected"
            )
            self.assertEqual(
                json.loads((output_dir / "adapter_config.json").read_text(encoding="utf-8")),
                {"selected": True},
            )

    def test_validation_is_a_supported_supervised_dataset_split(self) -> None:
        self.assertEqual(
            self.prepare_script._resolve_mode("validation", None), "supervised"
        )
        self.assertEqual(self.prepare_script._resolve_mode("test", None), "inference")
        with self.assertRaisesRegex(ValueError, "forbidden"):
            self.prepare_script._resolve_mode("test", "supervised")

    def test_training_script_has_no_test_data_input(self) -> None:
        source = (BASELINE_ROOT / "scripts" / "train_lora.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("test_jsonl", source)
        self.assertNotIn("test_mat", source)

    def test_configuration_requires_a_validation_mat_and_key(self) -> None:
        config = {
            "data": {
                "train_mat": "<TRAIN>",
                "validation_mat": "<VALIDATION>",
                "test_mat": "<TEST>",
                "train_key": "train_data",
                "validation_key": "validation_data",
                "test_key": "test_data",
            },
            "paths": {},
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            loaded = load_config(path)
            self.assertEqual(loaded["data"]["validation_mat"], "<VALIDATION>")

            del config["data"]["validation_key"]
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "validation_key"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
