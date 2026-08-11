"""Self-contained motion-query decoder for the adapted LLC-PC baseline.

This module is an independent, compact implementation of the mechanism used
by LLM-augmented motion-query predictors:

* observed agent histories and local lane polylines form decoder memory;
* fixed intention points provide spatial motion-query priors;
* four retrieved 17-dimensional semantic contexts initialize query content;
* a Transformer decoder produces multimodal future trajectories and scores.

It intentionally does not import or copy the WOMD-specific MTR codebase.  The
interface is small enough to run on the post-crash lane-changing tensors
created by :mod:`llc_pc.data_adapter` while preserving the method-level
conditioning mechanism needed by this baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .conditioning import ContextProjector, MotionQueryConditioner


@dataclass(frozen=True)
class LLCPCModelConfig:
    """Architecture settings for :class:`LLCPCMotionTransformer`."""

    agent_feature_dim: int = 7
    map_feature_dim: int = 2
    context_dim: int = 17
    context_window: int = 4
    d_model: int = 256
    nhead: int = 8
    agent_encoder_layers: int = 3
    decoder_layers: int = 3
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_history_steps: int = 10
    max_agents: int = 7
    future_steps: int = 50
    num_output_modes: int = 6


def _safe_temporal_mask(mask: Tensor) -> tuple[Tensor, Tensor]:
    """Make all-missing sequences safe for ``TransformerEncoder``.

    Returns a possibly adjusted valid mask and a boolean flag identifying
    sequences that were originally empty.  Empty sequences receive one zero
    token during attention and are reset to zero after pooling.
    """

    if mask.dtype is not torch.bool:
        mask = mask.bool()
    empty = ~mask.any(dim=-1)
    if empty.any():
        mask = mask.clone()
        mask[empty, 0] = True
    return mask, empty


class AgentHistoryEncoder(nn.Module):
    """Encode one-second histories for the ego vehicle and its neighbours."""

    def __init__(self, cfg: LLCPCModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.agent_feature_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
        )
        self.time_embedding = nn.Parameter(
            torch.zeros(cfg.max_history_steps, cfg.d_model)
        )
        self.role_embedding = nn.Embedding(cfg.max_agents, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=cfg.agent_encoder_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, histories: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Return agent tokens and an agent-level validity mask.

        Parameters
        ----------
        histories:
            ``[batch, agents, history, features]``.
        valid_mask:
            Boolean ``[batch, agents, history]``; missing neighbours are false.
        """

        if histories.ndim != 4:
            raise ValueError("histories must have shape [B, A, T, F]")
        batch, agents, steps, features = histories.shape
        if features != self.cfg.agent_feature_dim:
            raise ValueError(
                f"expected {self.cfg.agent_feature_dim} agent features, got {features}"
            )
        if steps > self.cfg.max_history_steps:
            raise ValueError("history is longer than max_history_steps")
        if agents > self.cfg.max_agents:
            raise ValueError("number of agents exceeds max_agents")
        if valid_mask.shape != histories.shape[:3]:
            raise ValueError("valid_mask must match histories[:3]")

        flat = histories.reshape(batch * agents, steps, features)
        mask = valid_mask.reshape(batch * agents, steps)
        safe_mask, empty = _safe_temporal_mask(mask)

        # NaNs are never allowed to enter attention.  Their corresponding mask
        # entries are false in the data adapter, but replacing them is still a
        # useful defence against accidental propagation.
        flat = torch.nan_to_num(flat)
        token = self.input_proj(flat)
        token = token + self.time_embedding[:steps].unsqueeze(0)
        token = self.encoder(token, src_key_padding_mask=~safe_mask)

        weights = safe_mask.to(token.dtype).unsqueeze(-1)
        pooled = (token * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        pooled[empty] = 0.0
        pooled = pooled.view(batch, agents, self.cfg.d_model)

        role_ids = torch.arange(agents, device=histories.device)
        pooled = pooled + self.role_embedding(role_ids).unsqueeze(0)
        pooled = self.output_norm(pooled)
        agent_valid = valid_mask.any(dim=-1)
        pooled = pooled * agent_valid.unsqueeze(-1).to(pooled.dtype)
        return pooled, agent_valid


class PolylineEncoder(nn.Module):
    """Encode a small set of locally reconstructed lane polylines."""

    def __init__(self, cfg: LLCPCModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.point_mlp = nn.Sequential(
            nn.Linear(cfg.map_feature_dim, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )
        self.no_map_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))

    def forward(self, polylines: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Return polyline tokens and validity, with a safe no-map fallback."""

        if polylines.ndim != 4:
            raise ValueError("polylines must have shape [B, P, L, F]")
        batch, count, points, features = polylines.shape
        if features != self.cfg.map_feature_dim:
            raise ValueError(
                f"expected {self.cfg.map_feature_dim} map features, got {features}"
            )
        if valid_mask.shape != polylines.shape[:3]:
            raise ValueError("map valid_mask must match polylines[:3]")

        if count == 0:
            return (
                self.no_map_token.expand(batch, 1, -1),
                torch.ones(batch, 1, dtype=torch.bool, device=polylines.device),
            )

        valid_mask = valid_mask.bool()
        token = self.point_mlp(torch.nan_to_num(polylines))
        neg_inf = torch.finfo(token.dtype).min
        token = token.masked_fill(~valid_mask.unsqueeze(-1), neg_inf)
        pooled = token.max(dim=2).values
        polyline_valid = valid_mask.any(dim=-1)
        pooled = torch.where(polyline_valid.unsqueeze(-1), pooled, torch.zeros_like(pooled))

        # TransformerDecoder requires at least one valid memory token.  Add a
        # learned no-map token only for scenes with no reconstructed lane.
        no_map = ~polyline_valid.any(dim=-1)
        if no_map.any():
            pooled = pooled.clone()
            polyline_valid = polyline_valid.clone()
            pooled[no_map, 0] = self.no_map_token[0, 0]
            polyline_valid[no_map, 0] = True
        return pooled, polyline_valid


class LLCPCMotionTransformer(nn.Module):
    """Compact MTR-style predictor with 17-D semantic query conditioning."""

    def __init__(self, cfg: LLCPCModelConfig, intention_points: Tensor) -> None:
        super().__init__()
        if intention_points.ndim != 2 or intention_points.shape[-1] != 2:
            raise ValueError("intention_points must have shape [Q, 2]")
        if intention_points.shape[0] % cfg.context_window != 0:
            raise ValueError("number of intention points must be divisible by context_window")
        if cfg.num_output_modes > intention_points.shape[0]:
            raise ValueError("num_output_modes cannot exceed number of intention points")

        self.cfg = cfg
        self.num_queries = int(intention_points.shape[0])
        self.register_buffer("intention_points", intention_points.float().clone())

        self.agent_encoder = AgentHistoryEncoder(cfg)
        self.map_encoder = PolylineEncoder(cfg)
        context_projector = ContextProjector(
            context_dim=cfg.context_dim,
            hidden_dim=cfg.d_model * 2,
            query_dim=cfg.d_model,
            activation="gelu",
            dropout=cfg.dropout,
        )
        self.query_conditioner = MotionQueryConditioner(
            projector=context_projector,
            assignment="cyclic",
            combine="replace",
        )
        self.intention_proj = nn.Sequential(
            nn.Linear(2, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, cfg.decoder_layers)
        self.query_norm = nn.LayerNorm(cfg.d_model)
        self.score_head = nn.Linear(cfg.d_model, 1)
        self.trajectory_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.future_steps * 2),
        )

    def set_intention_points(self, points: Tensor) -> None:
        """Replace anchors with training-split clusters of the same shape."""

        if points.shape != self.intention_points.shape:
            raise ValueError(
                f"expected intention points {tuple(self.intention_points.shape)}, "
                f"got {tuple(points.shape)}"
            )
        self.intention_points.copy_(points.to(self.intention_points))

    def _conditioned_queries(
        self, contexts: Tensor, context_mask: Optional[Tensor] = None
    ) -> Tensor:
        if contexts.ndim != 3:
            raise ValueError("contexts must have shape [B, W, 17]")
        batch, window, dim = contexts.shape
        if window != self.cfg.context_window or dim != self.cfg.context_dim:
            raise ValueError(
                f"expected contexts [B, {self.cfg.context_window}, "
                f"{self.cfg.context_dim}]"
            )
        semantic = self.query_conditioner(
            contexts,
            self.num_queries,
            context_mask=context_mask,
        )
        spatial = self.intention_proj(self.intention_points).unsqueeze(0)
        return self.query_norm(semantic + spatial)

    def forward(
        self,
        agent_histories: Tensor,
        agent_valid_mask: Tensor,
        map_polylines: Tensor,
        map_valid_mask: Tensor,
        semantic_contexts: Tensor,
        semantic_context_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Predict all query trajectories and top-scored output modes."""

        agent_tokens, agent_valid = self.agent_encoder(
            agent_histories, agent_valid_mask
        )
        map_tokens, map_valid = self.map_encoder(map_polylines, map_valid_mask)
        memory = torch.cat([agent_tokens, map_tokens], dim=1)
        memory_valid = torch.cat([agent_valid, map_valid], dim=1)

        queries = self._conditioned_queries(
            semantic_contexts, context_mask=semantic_context_mask
        )
        decoded = self.decoder(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=~memory_valid,
        )
        scores = self.score_head(decoded).squeeze(-1)
        residual = self.trajectory_head(decoded).view(
            decoded.shape[0], self.num_queries, self.cfg.future_steps, 2
        )

        fractions = torch.linspace(
            1.0 / self.cfg.future_steps,
            1.0,
            self.cfg.future_steps,
            device=decoded.device,
            dtype=decoded.dtype,
        )
        base = (
            self.intention_points[None, :, None, :]
            * fractions[None, None, :, None]
        )
        trajectories = base + residual

        top_scores, top_indices = scores.topk(
            k=self.cfg.num_output_modes, dim=-1, largest=True
        )
        gather_idx = top_indices[..., None, None].expand(
            -1, -1, self.cfg.future_steps, 2
        )
        top_trajectories = trajectories.gather(1, gather_idx)
        return {
            "scores": scores,
            "trajectories": trajectories,
            "top_scores": top_scores,
            "top_indices": top_indices,
            "top_trajectories": top_trajectories,
        }


def llc_pc_loss(
    prediction: Dict[str, Tensor],
    target: Tensor,
    valid_mask: Optional[Tensor] = None,
    classification_weight: float = 1.0,
    regression_weight: float = 1.0,
) -> Dict[str, Tensor]:
    """Winner-takes-all training objective using the closest final endpoint."""

    trajectories = prediction["trajectories"]
    scores = prediction["scores"]
    if target.ndim != 3 or target.shape[-1] != 2:
        raise ValueError("target must have shape [B, T, 2]")
    if trajectories.shape[0] != target.shape[0] or trajectories.shape[2:] != target.shape[1:]:
        raise ValueError("target shape must match predicted trajectory horizon")

    if valid_mask is None:
        valid_mask = torch.ones(
            target.shape[:2], dtype=torch.bool, device=target.device
        )
    if valid_mask.shape != target.shape[:2]:
        raise ValueError("valid_mask must have shape [B, T]")
    valid_mask = valid_mask.bool()
    if (~valid_mask.any(dim=-1)).any():
        raise ValueError("each target must contain at least one valid future step")

    last_index = valid_mask.long().sum(dim=-1) - 1
    batch_index = torch.arange(target.shape[0], device=target.device)
    gt_final = target[batch_index, last_index]
    pred_final = trajectories[batch_index, :, last_index, :]
    endpoint_distance = torch.linalg.vector_norm(
        pred_final - gt_final[:, None, :], dim=-1
    )
    assigned_query = endpoint_distance.argmin(dim=-1)

    chosen = trajectories[batch_index, assigned_query]
    element_loss = F.smooth_l1_loss(chosen, target, reduction="none").sum(dim=-1)
    regression = (
        element_loss * valid_mask.to(element_loss.dtype)
    ).sum() / valid_mask.sum().clamp_min(1)
    classification = F.cross_entropy(scores, assigned_query)
    total = regression_weight * regression + classification_weight * classification
    return {
        "loss": total,
        "regression_loss": regression,
        "classification_loss": classification,
        "assigned_query": assigned_query,
    }


def top1_trajectory(prediction: Dict[str, Tensor]) -> Tensor:
    """Return the highest-score trajectory for paper-aligned ADE/FDE."""

    scores = prediction["scores"]
    trajectories = prediction["trajectories"]
    index = scores.argmax(dim=-1)
    batch = torch.arange(scores.shape[0], device=scores.device)
    return trajectories[batch, index]


def ade_fde(
    prediction: Tensor, target: Tensor, valid_mask: Optional[Tensor] = None
) -> tuple[Tensor, Tensor]:
    """Compute mean top-1 ADE and FDE without oracle mode selection."""

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must both have shape [B, T, 2]")
    if valid_mask is None:
        valid_mask = torch.ones(
            target.shape[:2], dtype=torch.bool, device=target.device
        )
    valid_mask = valid_mask.bool()
    distance = torch.linalg.vector_norm(prediction - target, dim=-1)
    ade_per_sample = (
        distance * valid_mask.to(distance.dtype)
    ).sum(dim=-1) / valid_mask.sum(dim=-1).clamp_min(1)
    last_index = valid_mask.long().sum(dim=-1) - 1
    batch = torch.arange(target.shape[0], device=target.device)
    fde_per_sample = distance[batch, last_index]
    return ade_per_sample.mean(), fde_per_sample.mean()
