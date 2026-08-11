"""Training-only intention-point construction for LLC-PC motion queries."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


class IntentionPointKMeans:
    """Deterministic NumPy K-means over future endpoints in local coordinates.

    Future trajectories must already be transformed into the target vehicle's
    local frame.  Test futures must never be passed to :meth:`fit`.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        n_clusters: int = 64,
        *,
        random_state: int = 42,
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
    ) -> None:
        if n_clusters <= 0 or n_init <= 0 or max_iter <= 0 or tol < 0:
            raise ValueError("invalid K-means configuration")
        self.n_clusters = int(n_clusters)
        self.random_state = int(random_state)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.cluster_centers_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None
        self.n_iter_: Optional[int] = None
        self.source_split: Optional[str] = None
        self.num_training_endpoints_: Optional[int] = None

    def fit(
        self,
        future_trajectories: np.ndarray,
        *,
        future_mask: Optional[np.ndarray] = None,
        source_split: str = "train",
    ) -> "IntentionPointKMeans":
        """Fit anchors from the last valid training future point."""

        if str(source_split).strip().lower() != "train":
            raise ValueError("intention points may be fitted only on training futures")
        endpoints, _ = extract_future_endpoints(future_trajectories, future_mask)
        unique_count = np.unique(endpoints, axis=0).shape[0]
        if endpoints.shape[0] < self.n_clusters or unique_count < self.n_clusters:
            raise ValueError(
                "n_clusters cannot exceed the number of valid, unique training endpoints"
            )

        master_rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(self.n_init):
            run_seed = int(master_rng.integers(0, np.iinfo(np.int64).max))
            result = self._fit_once(endpoints, np.random.default_rng(run_seed))
            if best is None or result[1] < best[1]:
                best = result
        assert best is not None
        centers, inertia, iterations = best
        self.cluster_centers_ = centers.astype(np.float32)
        self.inertia_ = float(inertia)
        self.n_iter_ = int(iterations)
        self.source_split = "train"
        self.num_training_endpoints_ = int(endpoints.shape[0])
        return self

    def predict(self, endpoints: np.ndarray) -> np.ndarray:
        """Return the nearest anchor index for local-frame endpoints ``[N, 2]``."""

        self._require_fitted()
        points = np.asarray(endpoints, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("endpoints must be a finite [N, 2] array")
        assert self.cluster_centers_ is not None
        squared = _squared_distances(points, self.cluster_centers_)
        return np.argmin(squared, axis=1).astype(np.int64)

    def save(self, path: Union[str, Path]) -> None:
        self._require_fitted()
        assert self.cluster_centers_ is not None
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            format_version=np.asarray(self.FORMAT_VERSION, dtype=np.int64),
            source_split=np.asarray("train"),
            coordinate_frame=np.asarray("target_vehicle_local"),
            cluster_centers=self.cluster_centers_,
            n_clusters=np.asarray(self.n_clusters, dtype=np.int64),
            random_state=np.asarray(self.random_state, dtype=np.int64),
            n_init=np.asarray(self.n_init, dtype=np.int64),
            max_iter=np.asarray(self.max_iter, dtype=np.int64),
            tol=np.asarray(self.tol, dtype=np.float64),
            inertia=np.asarray(self.inertia_, dtype=np.float64),
            n_iter=np.asarray(self.n_iter_, dtype=np.int64),
            num_training_endpoints=np.asarray(self.num_training_endpoints_, dtype=np.int64),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "IntentionPointKMeans":
        with np.load(Path(path), allow_pickle=False) as data:
            if int(data["format_version"].item()) != cls.FORMAT_VERSION:
                raise ValueError("unsupported intention-point format version")
            if str(data["source_split"].item()) != "train":
                raise ValueError("refusing to load intention points not marked training-only")
            if str(data["coordinate_frame"].item()) != "target_vehicle_local":
                raise ValueError("intention points are not in the required local coordinate frame")
            model = cls(
                n_clusters=int(data["n_clusters"].item()),
                random_state=int(data["random_state"].item()),
                n_init=int(data["n_init"].item()),
                max_iter=int(data["max_iter"].item()),
                tol=float(data["tol"].item()),
            )
            centers = np.asarray(data["cluster_centers"], dtype=np.float32)
            if centers.shape != (model.n_clusters, 2) or not np.isfinite(centers).all():
                raise ValueError("invalid cluster centers in artifact")
            model.cluster_centers_ = centers
            model.inertia_ = float(data["inertia"].item())
            model.n_iter_ = int(data["n_iter"].item())
            model.num_training_endpoints_ = int(data["num_training_endpoints"].item())
            model.source_split = "train"
        return model

    def _fit_once(
        self, points: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, float, int]:
        centers = _kmeans_plus_plus(points, self.n_clusters, rng)
        for iteration in range(1, self.max_iter + 1):
            squared = _squared_distances(points, centers)
            labels = np.argmin(squared, axis=1)
            new_centers = np.empty_like(centers)
            nearest_squared = squared[np.arange(points.shape[0]), labels]
            for cluster in range(self.n_clusters):
                members = points[labels == cluster]
                if members.shape[0]:
                    new_centers[cluster] = members.mean(axis=0)
                else:
                    # Re-seed an empty cluster at the currently least represented point.
                    farthest = int(np.argmax(nearest_squared))
                    new_centers[cluster] = points[farthest]
                    nearest_squared[farthest] = -1.0
            shift = float(np.max(np.linalg.norm(new_centers - centers, axis=1)))
            centers = new_centers
            if shift <= self.tol:
                break
        final_squared = _squared_distances(points, centers)
        inertia = float(np.min(final_squared, axis=1).sum())
        return centers, inertia, iteration

    def _require_fitted(self) -> None:
        if self.cluster_centers_ is None:
            raise RuntimeError("intention-point K-means has not been fitted")


def extract_future_endpoints(
    future_trajectories: np.ndarray,
    future_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the last valid ``(x, y)`` point and retained source indices."""

    trajectories = np.asarray(future_trajectories, dtype=np.float32)
    if trajectories.ndim != 3 or trajectories.shape[1] == 0 or trajectories.shape[2] != 2:
        raise ValueError("future_trajectories must have shape [N, T, 2]")
    if future_mask is None:
        keep = np.isfinite(trajectories[:, -1, :]).all(axis=1)
        source_indices = np.flatnonzero(keep)
        endpoints = trajectories[source_indices, -1, :]
    else:
        mask = np.asarray(future_mask, dtype=bool)
        if mask.shape != trajectories.shape[:2]:
            raise ValueError("future_mask must have shape [N, T]")
        finite = np.isfinite(trajectories).all(axis=2)
        valid = mask & finite
        keep = valid.any(axis=1)
        source_indices = np.flatnonzero(keep)
        reverse_offset = np.argmax(valid[source_indices, ::-1], axis=1)
        final_steps = trajectories.shape[1] - 1 - reverse_offset
        endpoints = trajectories[source_indices, final_steps, :]
    if endpoints.shape[0] == 0:
        raise ValueError("no valid future endpoints were found")
    return endpoints.astype(np.float32), source_indices.astype(np.int64)


def _squared_distances(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.maximum(
        np.sum(points * points, axis=1, keepdims=True)
        + np.sum(centers * centers, axis=1)[None, :]
        - 2.0 * (points @ centers.T),
        0.0,
    )


def _kmeans_plus_plus(
    points: np.ndarray, n_clusters: int, rng: np.random.Generator
) -> np.ndarray:
    centers = np.empty((n_clusters, points.shape[1]), dtype=np.float32)
    first = int(rng.integers(points.shape[0]))
    centers[0] = points[first]
    closest = _squared_distances(points, centers[:1])[:, 0]
    for index in range(1, n_clusters):
        total = float(closest.sum())
        if total <= np.finfo(np.float32).eps:
            raise ValueError("not enough distinct endpoints for K-means++ initialization")
        selected = int(rng.choice(points.shape[0], p=closest / total))
        centers[index] = points[selected]
        closest = np.minimum(closest, _squared_distances(points, centers[index : index + 1])[:, 0])
    return centers
