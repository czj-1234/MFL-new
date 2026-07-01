from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


ArrayLike = np.ndarray | torch.Tensor


def _as_numpy_2d(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D array, got shape={x.shape}.")
    return x


def _orthonormalize(columns: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(columns)
    return q


@dataclass
class CounterfactualSubspace:
    basis: np.ndarray
    singular_values: np.ndarray
    mean_difference: np.ndarray
    rank: int

    def __post_init__(self) -> None:
        self.basis = np.asarray(self.basis, dtype=np.float64)
        self.singular_values = np.asarray(self.singular_values, dtype=np.float64)
        self.mean_difference = np.asarray(self.mean_difference, dtype=np.float64)
        if self.basis.ndim != 2:
            raise ValueError("basis must be a 2D array.")
        if self.basis.shape[1] != self.rank:
            raise ValueError("basis second dimension must equal rank.")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            basis=self.basis,
            singular_values=self.singular_values,
            mean_difference=self.mean_difference,
            rank=np.asarray([self.rank], dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CounterfactualSubspace":
        data = np.load(path)
        return cls(
            basis=data["basis"],
            singular_values=data["singular_values"],
            mean_difference=data["mean_difference"],
            rank=int(data["rank"][0]),
        )


def fit_counterfactual_subspace(
    associated_updates: ArrayLike,
    counterfactual_updates: ArrayLike,
    rank: int = 3,
    center: bool = True,
) -> CounterfactualSubspace:
    """Fit a low-rank basis from paired associated/counterfactual updates.

    Each row must correspond to the same shadow client and training round in both
    inputs. The method computes paired differences and applies SVD.
    """
    assoc = _as_numpy_2d(associated_updates)
    cf = _as_numpy_2d(counterfactual_updates)

    if assoc.shape != cf.shape:
        raise ValueError(
            "Paired update matrices must have identical shapes: "
            f"associated={assoc.shape}, counterfactual={cf.shape}."
        )
    if assoc.shape[0] < 2:
        raise ValueError("At least two paired updates are required.")

    differences = assoc - cf
    mean_difference = differences.mean(axis=0)
    matrix = differences - mean_difference if center else differences

    max_rank = min(matrix.shape)
    if not 1 <= rank <= max_rank:
        raise ValueError(f"rank must be in [1, {max_rank}], got {rank}.")

    _, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    basis = _orthonormalize(vt[:rank].T)

    return CounterfactualSubspace(
        basis=basis,
        singular_values=singular_values,
        mean_difference=mean_difference,
        rank=rank,
    )


def project_onto_subspace(updates: ArrayLike, basis: ArrayLike) -> np.ndarray:
    x = _as_numpy_2d(updates)
    u = _as_numpy_2d(basis)
    if x.shape[1] != u.shape[0]:
        raise ValueError(f"Dimension mismatch: updates={x.shape}, basis={u.shape}.")
    u = _orthonormalize(u)
    return (x @ u) @ u.T


def remove_subspace_component(
    updates: ArrayLike,
    basis: ArrayLike,
    alpha: float = 1.0,
) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
    x = _as_numpy_2d(updates)
    projected = project_onto_subspace(x, basis)
    return x - alpha * projected


def subspace_similarity(basis_a: ArrayLike, basis_b: ArrayLike) -> float:
    """Return mean squared canonical correlation in [0, 1]."""
    a = _orthonormalize(_as_numpy_2d(basis_a))
    b = _orthonormalize(_as_numpy_2d(basis_b))
    r = min(a.shape[1], b.shape[1])
    singular_values = np.linalg.svd(a.T @ b, compute_uv=False)[:r]
    return float(np.mean(singular_values**2))


def principal_angles_degrees(basis_a: ArrayLike, basis_b: ArrayLike) -> np.ndarray:
    a = _orthonormalize(_as_numpy_2d(basis_a))
    b = _orthonormalize(_as_numpy_2d(basis_b))
    singular_values = np.linalg.svd(a.T @ b, compute_uv=False)
    singular_values = np.clip(singular_values, -1.0, 1.0)
    return np.degrees(np.arccos(singular_values))


def removed_energy_ratio(
    updates: ArrayLike,
    basis: ArrayLike,
    alpha: float = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    x = _as_numpy_2d(updates)
    removed = alpha * project_onto_subspace(x, basis)
    numerator = np.sum(removed**2, axis=1)
    denominator = np.sum(x**2, axis=1) + eps
    return numerator / denominator
