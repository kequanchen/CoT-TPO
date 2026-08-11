"""Synthetic smoke tests for the self-contained LLC-PC motion decoder."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


BASELINE_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from llc_pc.model import (  # noqa: E402
    LLCPCModelConfig,
    LLCPCMotionTransformer,
    ade_fde,
    llc_pc_loss,
    top1_trajectory,
)


class MotionModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.cfg = LLCPCModelConfig(
            agent_feature_dim=7,
            map_feature_dim=2,
            context_dim=17,
            context_window=2,
            d_model=16,
            nhead=4,
            agent_encoder_layers=1,
            decoder_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            max_history_steps=4,
            max_agents=3,
            future_steps=5,
            num_output_modes=2,
        )
        anchors = torch.tensor(
            [[4.0, -1.0], [4.0, 1.0], [6.0, -1.0], [6.0, 1.0]]
        )
        self.model = LLCPCMotionTransformer(self.cfg, anchors)

    def _inputs(self):
        agents = torch.randn(2, 3, 4, 7)
        agent_mask = torch.ones(2, 3, 4, dtype=torch.bool)
        # The third agent in sample two is missing; NaNs must be masked safely.
        agents[1, 2] = float("nan")
        agent_mask[1, 2] = False
        maps = torch.randn(2, 2, 3, 2)
        map_mask = torch.ones(2, 2, 3, dtype=torch.bool)
        # Exercise the learned no-map fallback for one sample.
        maps[1] = float("nan")
        map_mask[1] = False
        contexts = torch.randn(2, 2, 17)
        context_mask = torch.tensor([[True, True], [True, False]])
        return agents, agent_mask, maps, map_mask, contexts, context_mask

    def test_forward_loss_and_metrics(self) -> None:
        output = self.model(*self._inputs())
        self.assertEqual(output["scores"].shape, (2, 4))
        self.assertEqual(output["trajectories"].shape, (2, 4, 5, 2))
        self.assertEqual(output["top_trajectories"].shape, (2, 2, 5, 2))
        self.assertTrue(torch.isfinite(output["trajectories"]).all())

        target = torch.randn(2, 5, 2)
        losses = llc_pc_loss(output, target)
        self.assertTrue(torch.isfinite(losses["loss"]))
        top1 = top1_trajectory(output)
        ade, fde = ade_fde(top1, target)
        self.assertTrue(torch.isfinite(ade))
        self.assertTrue(torch.isfinite(fde))

    def test_context_mask_changes_assignment_without_nan(self) -> None:
        inputs = self._inputs()
        first = self.model(*inputs)
        contexts = inputs[4].clone()
        contexts[:, 1] += 1000.0
        second = self.model(
            inputs[0], inputs[1], inputs[2], inputs[3], contexts, inputs[5]
        )
        # Sample two masks the modified second context, so its semantic query
        # initialization is unaffected (the model is deterministic in eval).
        self.model.eval()
        with torch.no_grad():
            first = self.model(*inputs)
            second = self.model(
                inputs[0], inputs[1], inputs[2], inputs[3], contexts, inputs[5]
            )
        self.assertTrue(torch.allclose(first["scores"][1], second["scores"][1]))


if __name__ == "__main__":
    unittest.main()
