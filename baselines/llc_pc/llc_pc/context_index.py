"""Leakage-safe nearest-neighbour retrieval for LLC-PC semantic contexts.

The index deliberately accepts only training-source records.  Validation and
test samples are queries, never index members.  IDs are stored as strings so
that MATLAB numeric IDs and Python string IDs have one stable representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np


ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]


@dataclass(frozen=True)
class QueryResult:
    """Neighbours returned by :meth:`TrainContextIndex.query`.

    Invalid/padded neighbours have index ``-1``, infinite distance, an empty
    sample ID, an all-zero context, and ``False`` in ``valid_mask``.
    """

    contexts: np.ndarray
    indices: np.ndarray
    distances: np.ndarray
    sample_ids: np.ndarray
    valid_mask: np.ndarray

    def mean_context(self, inverse_distance: bool = False, eps: float = 1e-6) -> np.ndarray:
        """Aggregate retrieved contexts while ignoring padded neighbours."""

        if inverse_distance:
            weights = np.where(
                self.valid_mask,
                1.0 / np.maximum(self.distances, eps),
                0.0,
            )
        else:
            weights = self.valid_mask.astype(np.float32)
        denom = weights.sum(axis=1, keepdims=True)
        weights = np.divide(weights, denom, out=np.zeros_like(weights), where=denom > 0)
        return np.einsum("bk,bkc->bc", weights, self.contexts).astype(np.float32)


class TrainContextIndex:
    """Exact KNN index whose members must come from the training split.

    Parameters
    ----------
    metric:
        ``"euclidean"`` (default) or ``"cosine"``.
    context_dim:
        Width of the encoded LLC-PC context.  The published schema has 17
        components (8 intentions, 4 affordances, and 5 scenarios).
    query_batch_size:
        Bounds the temporary query-by-index distance matrix.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        metric: str = "euclidean",
        context_dim: int = 17,
        query_batch_size: int = 256,
    ) -> None:
        if metric not in {"euclidean", "cosine"}:
            raise ValueError("metric must be 'euclidean' or 'cosine'")
        if context_dim <= 0 or query_batch_size <= 0:
            raise ValueError("context_dim and query_batch_size must be positive")
        self.metric = metric
        self.context_dim = int(context_dim)
        self.query_batch_size = int(query_batch_size)
        self._embeddings: Optional[np.ndarray] = None
        self._contexts: Optional[np.ndarray] = None
        self._sample_ids: Optional[np.ndarray] = None
        self._event_ids: Optional[np.ndarray] = None
        self.source_split: Optional[str] = None

    @property
    def is_fitted(self) -> bool:
        return self._embeddings is not None

    @property
    def size(self) -> int:
        return 0 if self._embeddings is None else int(self._embeddings.shape[0])

    @property
    def embedding_dim(self) -> int:
        self._require_fitted()
        assert self._embeddings is not None
        return int(self._embeddings.shape[1])

    def fit(
        self,
        embeddings: ArrayLike,
        contexts: ArrayLike,
        sample_ids: Sequence[object],
        *,
        event_ids: Optional[Sequence[object]] = None,
        source_split: str = "train",
    ) -> "TrainContextIndex":
        """Build the index from training records only.

        Requiring an explicit split tag does not infer provenance; it makes
        accidental validation/test indexing fail at the API boundary and is
        also persisted in saved artifacts for auditing.
        """

        if str(source_split).strip().lower() != "train":
            raise ValueError("LLC-PC context indices may be fitted only on the training split")
        emb = _as_float_matrix(embeddings, "embeddings")
        ctx = _as_float_matrix(contexts, "contexts")
        if ctx.shape[1] != self.context_dim:
            raise ValueError(f"contexts must have {self.context_dim} columns, got {ctx.shape[1]}")
        if emb.shape[0] != ctx.shape[0]:
            raise ValueError("embeddings and contexts must contain the same number of rows")
        ids = _as_string_vector(sample_ids, "sample_ids", emb.shape[0])
        events = None
        if event_ids is not None:
            events = _as_string_vector(event_ids, "event_ids", emb.shape[0])

        if self.metric == "cosine":
            emb = _unit_normalize(emb, "embeddings")
        self._embeddings = np.ascontiguousarray(emb, dtype=np.float32)
        self._contexts = np.ascontiguousarray(ctx, dtype=np.float32)
        self._sample_ids = ids
        self._event_ids = events
        self.source_split = "train"
        return self

    def query(
        self,
        embeddings: ArrayLike,
        k: int = 4,
        *,
        sample_ids: Optional[Sequence[object]] = None,
        event_ids: Optional[Sequence[object]] = None,
        exclude_self: bool = False,
        exclude_same_event: bool = False,
    ) -> QueryResult:
        """Retrieve contexts for arbitrary query records.

        ``exclude_same_event`` is useful for event-level evaluation and for
        stricter training diagnostics.  Rows with fewer than ``k`` eligible
        neighbours are padded and marked in ``valid_mask``.
        """

        self._require_fitted()
        if k <= 0:
            raise ValueError("k must be positive")
        assert self._embeddings is not None
        assert self._contexts is not None
        assert self._sample_ids is not None

        queries = _as_float_matrix(embeddings, "embeddings")
        if queries.shape[1] != self.embedding_dim:
            raise ValueError(
                f"query embeddings have width {queries.shape[1]}, expected {self.embedding_dim}"
            )
        if self.metric == "cosine":
            queries = _unit_normalize(queries, "query embeddings")

        query_ids = None
        if exclude_self:
            if sample_ids is None:
                raise ValueError("sample_ids are required when exclude_self=True")
            query_ids = _as_string_vector(sample_ids, "sample_ids", queries.shape[0])

        query_events = None
        if exclude_same_event:
            if self._event_ids is None:
                raise ValueError("the index has no event_ids for event exclusion")
            if event_ids is None:
                raise ValueError("event_ids are required when exclude_same_event=True")
            query_events = _as_string_vector(event_ids, "event_ids", queries.shape[0])

        batch = queries.shape[0]
        out_indices = np.full((batch, k), -1, dtype=np.int64)
        out_distances = np.full((batch, k), np.inf, dtype=np.float32)
        out_valid = np.zeros((batch, k), dtype=bool)

        for start in range(0, batch, self.query_batch_size):
            end = min(start + self.query_batch_size, batch)
            distances = self._distance(queries[start:end])
            if query_ids is not None:
                distances[query_ids[start:end, None] == self._sample_ids[None, :]] = np.inf
            if query_events is not None:
                assert self._event_ids is not None
                distances[query_events[start:end, None] == self._event_ids[None, :]] = np.inf

            for local_row, row in enumerate(distances):
                valid_candidates = np.flatnonzero(np.isfinite(row))
                take = min(k, valid_candidates.size)
                if take == 0:
                    continue
                if valid_candidates.size > take:
                    values = row[valid_candidates]
                    chosen = valid_candidates[np.argpartition(values, take - 1)[:take]]
                else:
                    chosen = valid_candidates
                # Index is the secondary key, making equal-distance output stable.
                chosen = chosen[np.lexsort((chosen, row[chosen]))]
                output_row = start + local_row
                out_indices[output_row, :take] = chosen
                out_distances[output_row, :take] = row[chosen]
                out_valid[output_row, :take] = True

        safe_indices = np.maximum(out_indices, 0)
        out_contexts = self._contexts[safe_indices].copy()
        out_contexts[~out_valid] = 0.0
        out_ids = self._sample_ids[safe_indices].copy()
        out_ids[~out_valid] = ""
        return QueryResult(
            contexts=out_contexts.astype(np.float32, copy=False),
            indices=out_indices,
            distances=out_distances,
            sample_ids=out_ids,
            valid_mask=out_valid,
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save a fitted training index without pickle-dependent objects."""

        self._require_fitted()
        assert self._embeddings is not None
        assert self._contexts is not None
        assert self._sample_ids is not None
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            format_version=np.asarray(self.FORMAT_VERSION, dtype=np.int64),
            source_split=np.asarray("train"),
            metric=np.asarray(self.metric),
            context_dim=np.asarray(self.context_dim, dtype=np.int64),
            query_batch_size=np.asarray(self.query_batch_size, dtype=np.int64),
            embeddings=self._embeddings,
            contexts=self._contexts,
            sample_ids=self._sample_ids.astype(str),
            has_event_ids=np.asarray(self._event_ids is not None),
            event_ids=(
                self._event_ids.astype(str)
                if self._event_ids is not None
                else np.asarray([], dtype="<U1")
            ),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TrainContextIndex":
        """Load and validate an index previously written by :meth:`save`."""

        with np.load(Path(path), allow_pickle=False) as data:
            if int(data["format_version"].item()) != cls.FORMAT_VERSION:
                raise ValueError("unsupported context-index format version")
            if str(data["source_split"].item()) != "train":
                raise ValueError("refusing to load an index not marked as training-only")
            index = cls(
                metric=str(data["metric"].item()),
                context_dim=int(data["context_dim"].item()),
                query_batch_size=int(data["query_batch_size"].item()),
            )
            event_ids = data["event_ids"] if bool(data["has_event_ids"].item()) else None
            index.fit(
                data["embeddings"],
                data["contexts"],
                data["sample_ids"],
                event_ids=event_ids,
                source_split="train",
            )
        return index

    def _distance(self, queries: np.ndarray) -> np.ndarray:
        assert self._embeddings is not None
        if self.metric == "cosine":
            return np.maximum(1.0 - queries @ self._embeddings.T, 0.0).astype(np.float32)
        q_sq = np.sum(queries * queries, axis=1, keepdims=True)
        i_sq = np.sum(self._embeddings * self._embeddings, axis=1)[None, :]
        squared = np.maximum(q_sq + i_sq - 2.0 * (queries @ self._embeddings.T), 0.0)
        return np.sqrt(squared).astype(np.float32)

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("context index has not been fitted")


def _as_float_matrix(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty rank-2 array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _as_string_vector(values: Sequence[object], name: str, length: int) -> np.ndarray:
    sequence = list(values)
    if len(sequence) != length:
        raise ValueError(f"{name} must contain {length} values")
    result = np.asarray([str(item) for item in sequence], dtype=str)
    if np.any(result == ""):
        raise ValueError(f"{name} must not contain empty IDs")
    return result


def _unit_normalize(values: np.ndarray, name: str) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float32).eps):
        raise ValueError(f"{name} contains a zero-norm row, invalid for cosine distance")
    return values / norms
