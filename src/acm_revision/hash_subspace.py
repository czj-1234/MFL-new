from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Mapping, MutableMapping, Tuple

import numpy as np
import torch

from .defenses import name_in_group

# Odd 64-bit multiplier. For power-of-two sketch sizes this visits each bucket
# exactly once per complete block of k consecutive coordinates.
_MULTIPLIER = np.uint64(11400714819323198485)
_UINT64_MASK = (1 << 64) - 1


def _stable_u64(*parts: object) -> int:
    payload = "|".join(str(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _validate_projection_dim(k: int) -> int:
    k = int(k)
    if k <= 0:
        raise ValueError("projection_dim must be positive")
    if k & (k - 1):
        raise ValueError("projection_dim must be a power of two for the orthonormal hash lift")
    return k


def group_dimension(delta: Mapping[str, torch.Tensor], group: str) -> int:
    return int(
        sum(
            int(tensor.numel())
            for name, tensor in delta.items()
            if torch.is_floating_point(tensor) and name_in_group(name, group)
        )
    )


def state_group_dimension(before: Mapping[str, torch.Tensor], group: str) -> int:
    return int(
        sum(
            int(tensor.numel())
            for name, tensor in before.items()
            if torch.is_floating_point(tensor) and name_in_group(name, group)
        )
    )


def _bucket_counts(dimension: int, k: int, seed64: int) -> np.ndarray:
    """Exact row counts for H when bucket(i)=(a*i+seed) mod k.

    Since a is odd and k is a power of two, every full block of k coordinates
    contains every bucket exactly once. This lets us normalize H so H H^T = I
    without scanning the full model merely to obtain bucket counts.
    """
    k = _validate_projection_dim(k)
    dimension = int(dimension)
    if dimension <= 0:
        return np.zeros(k, dtype=np.int64)
    base, rem = divmod(dimension, k)
    counts = np.full(k, base, dtype=np.int64)
    if rem:
        idx = np.arange(rem, dtype=np.uint64)
        hashed = idx * _MULTIPLIER + np.uint64(seed64)
        buckets = (hashed & np.uint64(k - 1)).astype(np.int64, copy=False)
        counts[buckets] += 1
    return counts


def _mapping(offset: int, size: int, k: int, seed64: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(int(offset), int(offset) + int(size), dtype=np.uint64)
    hashed = idx * _MULTIPLIER + np.uint64(seed64)
    buckets = (hashed & np.uint64(k - 1)).astype(np.int64, copy=False)
    signs = np.where((hashed >> np.uint64(63)) == 0, 1.0, -1.0).astype(np.float32, copy=False)
    return buckets, signs


def _group_seed(seed: int, group: str, dimension: int) -> int:
    return _stable_u64("mfl-orthonormal-hash", int(seed), group, int(dimension)) & _UINT64_MASK


def sketch_delta(
    delta: Mapping[str, torch.Tensor],
    group: str,
    projection_dim: int = 512,
    seed: int = 20260820,
    chunk_size: int = 1_048_576,
) -> np.ndarray:
    """Return z = H v for a row-orthonormal signed hash operator H.

    Every coordinate in the selected update group contributes. H is never
    materialized; only the k-dimensional sketch is kept in memory.
    """
    k = _validate_projection_dim(projection_dim)
    dimension = group_dimension(delta, group)
    if dimension <= 0:
        return np.zeros((0,), dtype=np.float32)
    seed64 = _group_seed(seed, group, dimension)
    counts = _bucket_counts(dimension, k, seed64)
    denom = np.sqrt(np.maximum(counts, 1)).astype(np.float64)
    out = np.zeros(k, dtype=np.float64)
    offset = 0
    for name in sorted(delta):
        tensor = delta[name]
        if not torch.is_floating_point(tensor) or not name_in_group(name, group):
            continue
        flat = tensor.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        for start in range(0, flat.size, int(chunk_size)):
            end = min(flat.size, start + int(chunk_size))
            buckets, signs = _mapping(offset + start, end - start, k, seed64)
            out += np.bincount(
                buckets,
                weights=flat[start:end].astype(np.float64, copy=False) * signs,
                minlength=k,
            )
        offset += flat.size
    if offset != dimension:
        raise AssertionError(f"Hash sketch consumed {offset}/{dimension} coordinates for {group}")
    out /= denom
    return out.astype(np.float32)


def sketch_state_delta(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    group: str,
    projection_dim: int = 512,
    seed: int = 20260820,
    chunk_size: int = 1_048_576,
) -> np.ndarray:
    """Hash an update directly from two state dicts without materializing delta."""
    k = _validate_projection_dim(projection_dim)
    dimension = state_group_dimension(before, group)
    if dimension <= 0:
        return np.zeros((0,), dtype=np.float32)
    seed64 = _group_seed(seed, group, dimension)
    counts = _bucket_counts(dimension, k, seed64)
    denom = np.sqrt(np.maximum(counts, 1)).astype(np.float64)
    out = np.zeros(k, dtype=np.float64)
    offset = 0
    for name in sorted(before):
        base = before[name]
        if not torch.is_floating_point(base) or not name_in_group(name, group):
            continue
        if name not in after:
            raise KeyError(f"Missing tensor {name} in after-state")
        left = base.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        right = after[name].detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        if left.size != right.size:
            raise ValueError(f"Shape mismatch for {name}")
        for start in range(0, left.size, int(chunk_size)):
            end = min(left.size, start + int(chunk_size))
            buckets, signs = _mapping(offset + start, end - start, k, seed64)
            values = right[start:end].astype(np.float64, copy=False) - left[start:end].astype(np.float64, copy=False)
            out += np.bincount(buckets, weights=values * signs, minlength=k)
        offset += left.size
    if offset != dimension:
        raise AssertionError(f"Hash sketch consumed {offset}/{dimension} coordinates for {group}")
    out /= denom
    return out.astype(np.float32)


def load_hash_basis(path: str | Path, group: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if group in data.files:
            basis = np.asarray(data[group], dtype=np.float32)
        elif "basis" in data.files:
            basis = np.asarray(data["basis"], dtype=np.float32)
        elif len(data.files) == 1:
            basis = np.asarray(data[data.files[0]], dtype=np.float32)
        else:
            raise KeyError(f"Could not select group={group!r} from {path}; keys={data.files}")
    if basis.ndim != 2:
        raise ValueError("Hash basis must be [projection_dim, rank]")
    return basis


def project_delta_inplace(
    delta: MutableMapping[str, torch.Tensor],
    group: str,
    basis: np.ndarray,
    alpha: float,
    projection_dim: int,
    seed: int,
    rank: int | None = None,
    chunk_size: int = 1_048_576,
) -> dict:
    """Apply v <- v - alpha B B^T v with B=H^T U.

    H is a row-orthonormal signed hash over *all* coordinates in the selected
    group and U is an orthonormal basis in hash space. Therefore B is an
    implicit orthonormal basis in the original update space. This implements a
    true low-rank full/layer update projector without storing D-by-r matrices.
    """
    k = _validate_projection_dim(projection_dim)
    dimension = group_dimension(delta, group)
    if dimension <= 0:
        return {"group": group, "skipped": True, "reason": "empty group"}
    basis = np.asarray(basis, dtype=np.float32)
    if basis.shape[0] != k:
        raise ValueError(f"Basis dimension {basis.shape[0]} != projection_dim {k}")
    if rank is not None:
        basis = basis[:, : min(int(rank), basis.shape[1])]
    if basis.shape[1] <= 0:
        raise ValueError("Basis rank must be positive")
    q, _ = np.linalg.qr(basis.astype(np.float64, copy=False))
    q = q.astype(np.float32)

    z = sketch_delta(delta, group, projection_dim=k, seed=seed, chunk_size=chunk_size)
    removed_z = q @ (q.T @ z)
    before_sketch_norm = float(np.linalg.norm(z))

    seed64 = _group_seed(seed, group, dimension)
    counts = _bucket_counts(dimension, k, seed64)
    inv_sqrt_counts = (1.0 / np.sqrt(np.maximum(counts, 1))).astype(np.float32)
    offset = 0
    alpha = float(alpha)
    for name in sorted(delta):
        tensor = delta[name]
        if not torch.is_floating_point(tensor) or not name_in_group(name, group):
            continue
        arr = tensor.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        for start in range(0, arr.size, int(chunk_size)):
            end = min(arr.size, start + int(chunk_size))
            buckets, signs = _mapping(offset + start, end - start, k, seed64)
            arr[start:end] -= alpha * signs * removed_z[buckets] * inv_sqrt_counts[buckets]
        offset += arr.size
    if offset != dimension:
        raise AssertionError(f"Hash backprojection consumed {offset}/{dimension} coordinates for {group}")

    after_z = z - alpha * removed_z
    return {
        "group": group,
        "name": "implicit_hash_filter",
        "original_dimension": int(dimension),
        "projection_dim": int(k),
        "rank": int(q.shape[1]),
        "alpha": alpha,
        "sketch_norm_before": before_sketch_norm,
        "sketch_norm_after": float(np.linalg.norm(after_z)),
        "all_group_coordinates_consumed": True,
        "implicit_original_space_basis": True,
    }


def clip_gaussian_delta_inplace(
    delta: MutableMapping[str, torch.Tensor],
    group: str,
    clip_norm: float,
    noise_l2_ratio: float,
    seed: int,
    chunk_size: int = 1_048_576,
) -> dict:
    """Full-group L2 clipping plus Gaussian noise without flattening the model.

    noise_l2_ratio specifies the expected noise L2 scale relative to clip_norm;
    per-coordinate std is noise_l2_ratio*clip_norm/sqrt(D).
    """
    clip_norm = float(clip_norm)
    noise_l2_ratio = float(noise_l2_ratio)
    if clip_norm <= 0 or noise_l2_ratio < 0:
        raise ValueError("clip_norm must be >0 and noise_l2_ratio must be >=0")
    dimension = group_dimension(delta, group)
    if dimension <= 0:
        return {"group": group, "skipped": True, "reason": "empty group"}

    sq = 0.0
    for name in sorted(delta):
        tensor = delta[name]
        if not torch.is_floating_point(tensor) or not name_in_group(name, group):
            continue
        arr = tensor.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        for start in range(0, arr.size, int(chunk_size)):
            block = arr[start : start + int(chunk_size)].astype(np.float64, copy=False)
            sq += float(np.dot(block, block))
    before_norm = math.sqrt(max(0.0, sq))
    scale = min(1.0, clip_norm / (before_norm + 1e-12))
    std = noise_l2_ratio * clip_norm / math.sqrt(float(dimension))
    rng = np.random.default_rng(int(seed))

    sq_after = 0.0
    for name in sorted(delta):
        tensor = delta[name]
        if not torch.is_floating_point(tensor) or not name_in_group(name, group):
            continue
        arr = tensor.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        for start in range(0, arr.size, int(chunk_size)):
            end = min(arr.size, start + int(chunk_size))
            arr[start:end] *= np.float32(scale)
            if std > 0:
                arr[start:end] += rng.normal(0.0, std, size=end - start).astype(np.float32)
            block = arr[start:end].astype(np.float64, copy=False)
            sq_after += float(np.dot(block, block))

    return {
        "group": group,
        "name": "streaming_clip_gaussian",
        "original_dimension": int(dimension),
        "clip_norm": clip_norm,
        "noise_l2_ratio": noise_l2_ratio,
        "per_coordinate_std": float(std),
        "before_norm": float(before_norm),
        "after_norm": float(math.sqrt(max(0.0, sq_after))),
        "all_group_coordinates_consumed": True,
    }
