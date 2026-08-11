"""Independent LLC-PC semantic-context conditioning for motion queries.

This module contains no upstream MTR source.  It exposes batch-first tensors
and an optional sequence-first view so it can be connected to an Apache-MTR
decoder or to the self-contained decoder released with this baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class MotionQueryInputs:
    """Prepared decoder inputs and the retrieved-context assignment map."""

    query_content: Tensor
    query_position: Tensor
    context_assignment: Tensor
    batch_first: bool


class ContextProjector(nn.Module):
    """Project a 17-component semantic context into decoder query space."""

    def __init__(
        self,
        context_dim: int = 17,
        hidden_dim: int = 512,
        query_dim: int = 256,
        *,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(context_dim, hidden_dim, query_dim) <= 0:
            raise ValueError("projection dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        activations = {"relu": nn.ReLU, "gelu": nn.GELU}
        if activation not in activations:
            raise ValueError("activation must be 'relu' or 'gelu'")
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        self.query_dim = int(query_dim)
        self.network = nn.Sequential(
            nn.Linear(self.context_dim, self.hidden_dim),
            activations[activation](),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.query_dim),
        )

    def forward(self, contexts: Tensor) -> Tensor:
        if contexts.ndim not in {2, 3} or contexts.shape[-1] != self.context_dim:
            raise ValueError(
                f"contexts must have shape [B, {self.context_dim}] or [B, K, {self.context_dim}]"
            )
        return self.network(contexts)


class MotionQueryConditioner(nn.Module):
    """Initialize motion-query content from retrieved semantic contexts.

    With ``assignment='cyclic'``, the valid KNN contexts for each sample are
    assigned to Q motion queries in repeated order.  Query positions remain a
    separate input; they normally encode the training-derived intention points.
    """

    def __init__(
        self,
        projector: Optional[ContextProjector] = None,
        *,
        assignment: str = "cyclic",
        combine: str = "replace",
        context_dim: int = 17,
        hidden_dim: int = 512,
        query_dim: int = 256,
    ) -> None:
        super().__init__()
        if assignment != "cyclic":
            raise ValueError("only deterministic cyclic assignment is supported")
        if combine not in {"replace", "add"}:
            raise ValueError("combine must be 'replace' or 'add'")
        self.projector = projector or ContextProjector(
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            query_dim=query_dim,
        )
        self.assignment = assignment
        self.combine = combine

    @property
    def query_dim(self) -> int:
        return self.projector.query_dim

    @property
    def context_dim(self) -> int:
        return self.projector.context_dim

    def forward(
        self,
        contexts: Tensor,
        num_queries: int,
        *,
        context_mask: Optional[Tensor] = None,
        base_query_content: Optional[Tensor] = None,
        return_assignment: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Return conditioned query content with shape ``[B, Q, D]``."""

        if num_queries <= 0:
            raise ValueError("num_queries must be positive")
        if contexts.ndim == 2:
            contexts = contexts.unsqueeze(1)
        if contexts.ndim != 3 or contexts.shape[-1] != self.context_dim:
            raise ValueError(f"contexts must have shape [B, K, {self.context_dim}]")
        batch, neighbours, _ = contexts.shape
        if neighbours == 0:
            raise ValueError("contexts must include at least one neighbour slot")
        if context_mask is None:
            context_mask = torch.ones(
                (batch, neighbours), dtype=torch.bool, device=contexts.device
            )
        else:
            if context_mask.shape != (batch, neighbours):
                raise ValueError("context_mask must have shape [B, K]")
            context_mask = context_mask.to(device=contexts.device, dtype=torch.bool)

        projected = self.projector(contexts)
        assignment = torch.full(
            (batch, num_queries), -1, dtype=torch.long, device=contexts.device
        )
        semantic_content = projected.new_zeros((batch, num_queries, self.query_dim))
        for batch_index in range(batch):
            valid = torch.nonzero(context_mask[batch_index], as_tuple=False).flatten()
            if valid.numel() == 0:
                continue
            selected = valid[torch.arange(num_queries, device=contexts.device) % valid.numel()]
            assignment[batch_index] = selected
            semantic_content[batch_index] = projected[batch_index, selected]

        base = _broadcast_query_content(
            base_query_content,
            batch=batch,
            num_queries=num_queries,
            query_dim=self.query_dim,
            reference=semantic_content,
        )
        assigned = assignment >= 0
        if self.combine == "add":
            output = base + semantic_content
        else:
            output = torch.where(assigned.unsqueeze(-1), semantic_content, base)
        if return_assignment:
            return output, assignment
        return output

    def prepare(
        self,
        contexts: Tensor,
        query_position: Tensor,
        *,
        context_mask: Optional[Tensor] = None,
        base_query_content: Optional[Tensor] = None,
        sequence_first: bool = False,
    ) -> MotionQueryInputs:
        """Prepare content/position tensors for a motion decoder.

        ``query_position`` may be ``[Q, D]`` (shared across a batch) or
        ``[B, Q, D]``.  An Apache-MTR adapter can request sequence-first
        ``[Q, B, D]`` tensors with ``sequence_first=True``.
        """

        if contexts.ndim == 2:
            batch = contexts.shape[0]
        elif contexts.ndim == 3:
            batch = contexts.shape[0]
        else:
            raise ValueError("contexts must be rank 2 or 3")
        position = _broadcast_positions(query_position, batch, self.query_dim)
        content, assignment = self.forward(
            contexts,
            position.shape[1],
            context_mask=context_mask,
            base_query_content=base_query_content,
            return_assignment=True,
        )
        if sequence_first:
            return MotionQueryInputs(
                query_content=content.transpose(0, 1).contiguous(),
                query_position=position.transpose(0, 1).contiguous(),
                context_assignment=assignment.transpose(0, 1).contiguous(),
                batch_first=False,
            )
        return MotionQueryInputs(content, position, assignment, True)


def _broadcast_query_content(
    values: Optional[Tensor],
    *,
    batch: int,
    num_queries: int,
    query_dim: int,
    reference: Tensor,
) -> Tensor:
    if values is None:
        return reference.new_zeros((batch, num_queries, query_dim))
    values = values.to(device=reference.device, dtype=reference.dtype)
    if values.shape == (num_queries, query_dim):
        return values.unsqueeze(0).expand(batch, -1, -1)
    if values.shape == (batch, num_queries, query_dim):
        return values
    raise ValueError("base_query_content must have shape [Q, D] or [B, Q, D]")


def _broadcast_positions(values: Tensor, batch: int, query_dim: int) -> Tensor:
    if values.ndim == 2 and values.shape[1] == query_dim:
        return values.unsqueeze(0).expand(batch, -1, -1)
    if values.ndim == 3 and values.shape[0] == batch and values.shape[2] == query_dim:
        return values
    raise ValueError("query_position must have shape [Q, D] or [B, Q, D]")
