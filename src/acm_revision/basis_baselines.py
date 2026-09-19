from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .attacks import load_matrix, read_metadata


def _orthonormal_columns(matrix: np.ndarray, rank: Optional[int] = None) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.size == 0:
        raise ValueError("Cannot build a basis from an empty matrix.")
    q, _ = np.linalg.qr(matrix)
    if rank is not None:
        q = q[:, : min(int(rank), q.shape[1])]
    return q.astype(np.float32)


def fit_pca_basis(
    metadata_paths: Sequence[str | Path],
    group: str,
    output_path: str | Path,
    rank: int,
    observation: str = "raw",
) -> dict:
    """Rank-matched unsupervised PCA baseline fitted on shadow data only."""
    frame = read_metadata(metadata_paths)
    x = load_matrix(frame, group, observation)
    max_rank = min(int(rank), x.shape[0] - 1, x.shape[1])
    if max_rank < 1:
        raise ValueError("Not enough shadow updates to fit PCA.")
    pca = PCA(n_components=max_rank, svd_solver="randomized", random_state=42)
    pca.fit(x)
    basis = pca.components_.T.astype(np.float32)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, basis=basis)
    meta = {
        "method": "rank_matched_pca",
        "fit_population": sorted(frame["population"].astype(str).unique().tolist()) if "population" in frame else [],
        "group": group,
        "observation": observation,
        "rank": int(max_rank),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "target_data_used": False,
    }
    with path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def fit_random_basis(
    dimension: int,
    output_path: str | Path,
    rank: int,
    seed: int = 42,
) -> dict:
    """Rank-matched random-subspace control."""
    if not 1 <= int(rank) <= int(dimension):
        raise ValueError("rank must be between 1 and dimension.")
    rng = np.random.default_rng(seed)
    basis = _orthonormal_columns(rng.normal(size=(int(dimension), int(rank))), rank=rank)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, basis=basis)
    meta = {
        "method": "rank_matched_random_subspace",
        "dimension": int(dimension),
        "rank": int(rank),
        "seed": int(seed),
        "target_data_used": False,
    }
    with path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def fit_supervised_sensitive_basis(
    metadata_paths: Sequence[str | Path],
    group: str,
    output_path: str | Path,
    rank: int,
    target: str = "dominant_label",
    observation: str = "raw",
    seed: int = 42,
) -> dict:
    """Supervised sensitive-direction removal baseline.

    A multinomial logistic attacker is fitted on standardized shadow updates.
    Its coefficient span is mapped back to the original update coordinates and
    orthonormalized. For binary tasks, a class-mean difference is added so rank
    can exceed the single logistic direction when requested.
    """
    frame = read_metadata(metadata_paths)
    if target not in frame.columns:
        raise KeyError(f"Target column {target} missing from metadata.")
    x = load_matrix(frame, group, observation)
    y = frame[target].to_numpy()
    scaler = StandardScaler(with_mean=True, with_std=True)
    x_scaled = scaler.fit_transform(x)
    model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=seed,
        multi_class="auto",
    )
    model.fit(x_scaled, y)
    directions = []
    coef = np.asarray(model.coef_, dtype=np.float64)
    # Transform standardized-space coefficients back into raw coordinates.
    raw_coef = coef / (np.asarray(scaler.scale_, dtype=np.float64)[None, :] + 1e-12)
    directions.extend(raw_coef)

    classes = sorted(np.unique(y).tolist())
    global_mean = x.mean(axis=0)
    for cls in classes:
        directions.append(x[y == cls].mean(axis=0) - global_mean)
    matrix = np.stack(directions, axis=1)
    basis = _orthonormal_columns(matrix, rank=rank)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, basis=basis)
    meta = {
        "method": "supervised_sensitive_direction",
        "fit_population": sorted(frame["population"].astype(str).unique().tolist()) if "population" in frame else [],
        "group": group,
        "observation": observation,
        "target": target,
        "rank_requested": int(rank),
        "rank_stored": int(basis.shape[1]),
        "target_data_used": False,
    }
    with path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def fit_gradient_orthogonalization_basis(
    metadata_paths: Sequence[str | Path],
    group: str,
    output_path: str | Path,
    rank: int,
    observation: str = "raw",
) -> dict:
    """Class-mean gradient/update direction baseline used for orthogonalization."""
    frame = read_metadata(metadata_paths)
    x = load_matrix(frame, group, observation)
    y = frame["dominant_label"].to_numpy()
    directions = []
    global_mean = x.mean(axis=0)
    for cls in sorted(np.unique(y).tolist()):
        directions.append(x[y == cls].mean(axis=0) - global_mean)
    basis = _orthonormal_columns(np.stack(directions, axis=1), rank=rank)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, basis=basis)
    meta = {
        "method": "gradient_orthogonalization",
        "group": group,
        "observation": observation,
        "rank_requested": int(rank),
        "rank_stored": int(basis.shape[1]),
        "target_data_used": False,
    }
    with path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta
