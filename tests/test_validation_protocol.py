from __future__ import annotations

import importlib.util
import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"protocol_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ValidationProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.student = load_script("train_strategy_student_mlp")
        cls.trajectory = load_script("train_cot_tp_film")
        cls.teacher = load_script("generate_llm_teacher")
        cls.parser = load_script("parse_strategy_vectors")

    def test_teacher_all_samples_processes_numpy_split_in_row_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path = root / "split.mat"
            data_path.touch()
            output_dir = root / "teacher"
            args = argparse.Namespace(
                data_path=data_path,
                data_key="validation_data",
                all_samples=True,
                random_sample=False,
                sample_idx=10,
                vehicle_length=4.8,
                vehicle_width=1.8,
                skip_llm=True,
                api_key_env="LLM_API_KEY",
                base_url=None,
                model="teacher",
                temperature=0.3,
                max_tokens=200,
                output_dir=output_dir,
                save_prompt=True,
                enable_thinking=False,
                stream=True,
            )
            processed: list[int] = []

            def process(_sample, index, **_kwargs):
                processed.append(index)
                return {"success": True, "skipped_llm": True}

            with (
                mock.patch.object(
                    self.teacher,
                    "load_mat_data",
                    return_value=np.asarray([{}, {}], dtype=object),
                ),
                mock.patch.object(
                    self.teacher, "build_column_schemas", return_value=(object(), object())
                ),
                mock.patch.object(
                    self.teacher, "parse_sample", return_value={"status": 0}
                ),
                mock.patch.object(self.teacher, "process_sample", side_effect=process),
            ):
                self.teacher.main(args)

            self.assertEqual(processed, [0, 1])
            summary = json.loads(
                (output_dir / "batch_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual([row["sample_idx"] for row in summary], [0, 1])

    def test_teacher_batch_rejects_stale_sample_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            (output_dir / "response_sample_0.txt").write_text("stale", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "prior sample artifacts"):
                self.teacher.require_clean_batch_output(output_dir)

    def test_empty_parse_removes_stale_vector_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            responses = root / "responses"
            responses.mkdir()
            output = root / "vectors"
            output.mkdir()
            np.save(output / "ids.npy", np.asarray([0], dtype=np.int32))
            np.save(output / "c.npy", np.ones((1, 2), dtype=np.float32))
            args = argparse.Namespace(
                responses_dir=responses,
                prompts_dir=None,
                out_dir=output,
                recursive=False,
                fallback_phase="anticipation",
                strict=True,
            )
            with mock.patch.object(self.parser, "parse_args", return_value=args):
                self.assertEqual(self.parser.main(), 0)
            self.assertFalse((output / "ids.npy").exists())
            self.assertFalse((output / "c.npy").exists())

    def test_test_loading_occurs_only_after_epoch_loop(self) -> None:
        for filename in (
            "train_strategy_student_mlp.py",
            "train_cot_tp_film.py",
        ):
            source = (ROOT / "scripts" / filename).read_text(encoding="utf-8")
            epoch_loop = source.index("for epoch in range(1, args.epochs + 1):")
            final_test = source.index("Loading the test set for one final evaluation")
            self.assertGreater(final_test, epoch_loop)
            self.assertNotIn("test_loader", source[epoch_loop:final_test])

    def test_strategy_vectors_require_exact_zero_based_coverage(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero-based MAT indices"):
            self.student.require_vector_coverage({1: np.zeros(2)}, 1, "training")
        with self.assertRaisesRegex(ValueError, "zero-based MAT indices"):
            self.trajectory.require_vector_coverage(
                {1: np.zeros(2)}, 1, "validation"
            )

    def test_duplicate_vector_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            vector_dir = Path(temporary)
            np.save(vector_dir / "ids.npy", np.asarray([0, 0], dtype=np.int32))
            np.save(vector_dir / "c.npy", np.zeros((2, 3), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "unique"):
                self.student.load_teacher_vectors(vector_dir)
            with self.assertRaisesRegex(ValueError, "unique"):
                self.trajectory.load_llm_vectors(vector_dir)

    def test_paper_teacher_defaults_to_qwen3_thinking_stream(self) -> None:
        with mock.patch.object(sys, "argv", ["generate_llm_teacher.py"]):
            args = self.teacher.parse_args()
        self.assertEqual(args.model, "qwen3-235b-a22b")
        self.assertTrue(args.enable_thinking)
        self.assertTrue(args.stream)

        with mock.patch.object(
            sys,
            "argv",
            ["generate_llm_teacher.py", "--disable-thinking", "--no-stream"],
        ):
            disabled = self.teacher.parse_args()
        self.assertFalse(disabled.enable_thinking)
        self.assertFalse(disabled.stream)

    def test_qwen3_streaming_response_is_collected_and_parsed(self) -> None:
        chunks = [
            SimpleNamespace(
                model="qwen3-235b-a22b",
                usage=None,
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(reasoning_content="reasoning ", content=None)
                    )
                ],
            ),
            SimpleNamespace(
                model="qwen3-235b-a22b",
                usage=None,
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(reasoning_content=None, content='{"status":')
                    )
                ],
            ),
            SimpleNamespace(
                model="qwen3-235b-a22b",
                usage=None,
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(reasoning_content=None, content=' "ok"}')
                    )
                ],
            ),
            SimpleNamespace(
                model="qwen3-235b-a22b",
                usage=SimpleNamespace(total_tokens=12),
                choices=[],
            ),
        ]
        create = mock.Mock(return_value=iter(chunks))
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        result = self.teacher.call_llm(
            "prompt",
            client,
            "qwen3-235b-a22b",
            temperature=0.3,
            max_tokens=2000,
            enable_thinking=True,
            stream=True,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["parsed_result"], {"status": "ok"})
        self.assertEqual(result["reasoning_content"], "reasoning ")
        request = create.call_args.kwargs
        self.assertTrue(request["stream"])
        self.assertEqual(request["extra_body"], {"enable_thinking": True})

    def test_non_thinking_request_explicitly_disables_thinking(self) -> None:
        response = SimpleNamespace(
            model="qwen3-235b-a22b",
            usage=None,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"status": "ok"}', reasoning_content=None)
                )
            ],
        )
        create = mock.Mock(return_value=response)
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        result = self.teacher.call_llm(
            "prompt",
            client,
            "qwen3-235b-a22b",
            temperature=0.3,
            max_tokens=2000,
            enable_thinking=False,
            stream=False,
        )

        self.assertTrue(result["success"])
        self.assertEqual(
            create.call_args.kwargs["extra_body"], {"enable_thinking": False}
        )

    def test_strategy_student_remains_frozen_downstream(self) -> None:
        torch = self.trajectory.torch
        student = self.trajectory.StrategyStudent(in_dim=4, hid_dim=8, out_dim=3)
        self.trajectory.freeze_student(student)

        self.assertFalse(student.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in student.parameters()))

        model = torch.nn.Linear(4, 2)
        optimizer = self.trajectory.build_optimizer(
            model,
            argparse.Namespace(
                lr=1e-4,
                film_lr=1.5e-4,
                weight_decay=1e-4,
                film_weight_decay=0.0,
            ),
        )
        optimizer_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        student_ids = {id(parameter) for parameter in student.parameters()}
        self.assertTrue(optimizer_ids.isdisjoint(student_ids))

        source = (ROOT / "scripts" / "train_cot_tp_film.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("unfreezing StrategyStudent", source)
        self.assertNotIn("requires_grad_(True)", source)
        self.assertNotIn("phase1_epochs", source)

    def test_frozen_student_requires_a_checkpoint(self) -> None:
        student = self.trajectory.StrategyStudent(in_dim=4, hid_dim=8, out_dim=3)
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing.pth"
            with self.assertRaisesRegex(FileNotFoundError, "Student checkpoint not found"):
                self.trajectory.load_student_checkpoint(
                    student, missing, self.trajectory.torch.device("cpu")
                )


if __name__ == "__main__":
    unittest.main()
