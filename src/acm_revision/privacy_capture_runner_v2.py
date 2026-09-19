from __future__ import annotations

import errno
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch

from . import privacy_capture_runner as base
from .defenses import flatten_delta


def _feature_hash_projection_pair(
    raw_delta: Mapping[str, torch.Tensor],
    observed_delta: Mapping[str, torch.Tensor],
    group: str,
    projection_dim: int,
    seed: int,
    chunk_size: int = 262_144,
):
    """Signed feature-hash projection that consumes every coordinate in a group.

    Unlike coordinate subsampling, every parameter update contributes to the
    projected feature. The mapping is deterministic across shadow and target
    runs and can therefore be used as a compressed full-update observation.
    """
    names, sizes = base._matching_layout(observed_delta, group)
    original_dim = int(sum(sizes))
    if original_dim <= 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), 0

    k = min(int(projection_dim), original_dim)
    if k <= 0:
        raise ValueError("projection_dim must be positive")

    raw_out = np.zeros(k, dtype=np.float64)
    obs_out = np.zeros(k, dtype=np.float64)
    group_seed = np.uint64(base._stable_seed(seed, group, original_dim))
    multiplier = np.uint64(11400714819323198485)
    offset = 0

    for name, size in zip(names, sizes):
        raw_flat = raw_delta[name].detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        obs_flat = observed_delta[name].detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        if raw_flat.size != size or obs_flat.size != size:
            raise AssertionError(f"Unexpected tensor size for {name}")

        for start in range(0, size, chunk_size):
            end = min(size, start + chunk_size)
            indices = np.arange(offset + start, offset + end, dtype=np.uint64)
            hashed = indices * multiplier + group_seed
            buckets = (hashed % np.uint64(k)).astype(np.int64, copy=False)
            signs = np.where((hashed >> np.uint64(63)) == 0, 1.0, -1.0)
            raw_out += np.bincount(
                buckets,
                weights=raw_flat[start:end].astype(np.float64, copy=False) * signs,
                minlength=k,
            )
            obs_out += np.bincount(
                buckets,
                weights=obs_flat[start:end].astype(np.float64, copy=False) * signs,
                minlength=k,
            )
        offset += size

    if offset != original_dim:
        raise AssertionError(f"Projection consumed {offset}/{original_dim} coordinates")
    return raw_out.astype(np.float32), obs_out.astype(np.float32), original_dim


def _local_spool_dir() -> Path:
    """Return a node-local spool directory for building NPZ archives.

    Results often live on NFS. Building ZIP archives directly on NFS keeps an
    open remote file handle for the whole compression operation and can fail
    with errno 116 (ESTALE). Compression is therefore completed on local disk
    first, then the finished archive is copied to the result filesystem.
    """
    base_dir = Path(os.environ.get("MFL_LOCAL_TMPDIR", tempfile.gettempdir()))
    spool = base_dir / f"mfl_npz_spool_{os.getuid()}"
    spool.mkdir(parents=True, exist_ok=True)
    return spool


def _copy_with_fsync(source: Path, destination: Path) -> None:
    with source.open("rb") as src, destination.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())


def _validate_npz(path: Path) -> None:
    if not path.exists() or path.stat().st_size <= 128:
        raise OSError(errno.EIO, f"Capture archive is missing or too small: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "__capture_info_json" not in archive.files:
            raise OSError(errno.EIO, f"Capture archive is incomplete: {path}")


def _robust_savez_compressed(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    """Save an NPZ safely when the result directory may be on NFS.

    The archive is compressed entirely on node-local storage. The completed
    file is then copied to a uniquely named sibling and atomically renamed.
    ESTALE/EIO/EAGAIN/ETIMEDOUT errors are retried with a fresh destination
    handle. A failed attempt never leaves the final path looking complete.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    spool = _local_spool_dir()
    fd, temp_name = tempfile.mkstemp(prefix="capture_", suffix=".npz", dir=spool)
    os.close(fd)
    local_tmp = Path(temp_name)

    try:
        np.savez_compressed(local_tmp, **arrays)
        _validate_npz(local_tmp)

        retryable = {
            getattr(errno, "ESTALE", 116),
            errno.EIO,
            errno.EAGAIN,
            getattr(errno, "ETIMEDOUT", 110),
        }
        max_attempts = int(os.environ.get("MFL_NFS_WRITE_RETRIES", "6"))
        last_error: OSError | None = None

        for attempt in range(1, max_attempts + 1):
            partial = path.with_name(
                f".{path.name}.partial.{os.getpid()}.{attempt}"
            )
            try:
                partial.unlink(missing_ok=True)
                _copy_with_fsync(local_tmp, partial)
                os.replace(partial, path)
                _validate_npz(path)
                return
            except OSError as exc:
                last_error = exc
                try:
                    partial.unlink(missing_ok=True)
                except OSError:
                    pass
                if exc.errno not in retryable or attempt >= max_attempts:
                    raise
                delay = min(30, 2 ** (attempt - 1))
                print(
                    f"[NFS WRITE RETRY] attempt={attempt}/{max_attempts} "
                    f"errno={exc.errno} path={path} sleep={delay}s",
                    flush=True,
                )
                time.sleep(delay)
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                except OSError:
                    pass

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Failed to save capture archive: {path}")
    finally:
        try:
            local_tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _save_capture_npz_factory_v2(cfg: dict):
    capture_cfg = cfg.get("privacy_capture", {})
    every_groups = list(
        capture_cfg.get("every_round_exact_groups", base.DEFAULT_EVERY_ROUND_GROUPS)
    )
    milestone_exact = list(
        capture_cfg.get("milestone_exact_groups", base.DEFAULT_MILESTONE_EXACT_GROUPS)
    )
    milestone_large = list(
        capture_cfg.get("milestone_sketch_groups", base.DEFAULT_MILESTONE_SKETCH_GROUPS)
    )
    projection_groups = set(capture_cfg.get("full_group_projection_groups", ["full_update"]))
    milestone_rounds = {
        int(x) for x in capture_cfg.get("milestone_rounds", base.DEFAULT_MILESTONE_ROUNDS)
    }
    sketch_dim = int(capture_cfg.get("sketch_dim", 16384))
    projection_dim = int(capture_cfg.get("full_group_projection_dim", 2048))
    sketch_seed = int(capture_cfg.get("sketch_seed", 20260803))
    projection_seed = int(capture_cfg.get("full_group_projection_seed", 20260804))

    def save_capture_npz(
        path: Path,
        raw_delta: Mapping[str, torch.Tensor],
        observed_delta: Mapping[str, torch.Tensor],
        groups: Sequence[str],
        storage_dtype: str,
    ) -> dict:
        del groups
        round_id = base._round_from_path(path)
        exact_groups = list(every_groups)
        large_groups: list[str] = []
        if round_id in milestone_rounds:
            exact_groups.extend(g for g in milestone_exact if g not in exact_groups)
            large_groups = list(milestone_large)

        dtype = np.float16 if storage_dtype == "float16" else np.float32
        arrays: Dict[str, np.ndarray] = {}
        dimensions: Dict[str, int] = {}
        representations: Dict[str, dict] = {}

        for group in exact_groups:
            raw, _ = flatten_delta(raw_delta, group)
            observed, _ = flatten_delta(observed_delta, group)
            if observed.size == 0:
                continue
            arrays[f"raw__{group}"] = raw.astype(dtype, copy=False)
            arrays[f"observed__{group}"] = observed.astype(dtype, copy=False)
            dimensions[group] = int(observed.size)
            representations[group] = {
                "representation": "exact",
                "feature_dim": int(observed.size),
                "original_dim": int(observed.size),
            }

        for group in large_groups:
            if group in projection_groups:
                raw, observed, original_dim = _feature_hash_projection_pair(
                    raw_delta,
                    observed_delta,
                    group,
                    projection_dim=projection_dim,
                    seed=projection_seed,
                )
                representation = "deterministic_signed_feature_hash_full_group"
                representation_seed = projection_seed
            else:
                raw, observed, original_dim = base._coordinate_sketch_pair(
                    raw_delta,
                    observed_delta,
                    group,
                    sketch_dim=sketch_dim,
                    seed=sketch_seed,
                )
                representation = "deterministic_coordinate_sketch"
                representation_seed = sketch_seed
            if observed.size == 0:
                continue
            arrays[f"raw__{group}"] = raw.astype(dtype, copy=False)
            arrays[f"observed__{group}"] = observed.astype(dtype, copy=False)
            dimensions[group] = int(observed.size)
            representations[group] = {
                "representation": representation,
                "feature_dim": int(observed.size),
                "original_dim": int(original_dim),
                "representation_seed": int(representation_seed),
                "all_group_coordinates_consumed": bool(group in projection_groups),
            }

        if not arrays:
            raise RuntimeError(f"No update arrays selected for {path}")

        capture_info = {
            "round": round_id,
            "milestone": round_id in milestone_rounds,
            "groups": representations,
            "storage_dtype": storage_dtype,
        }
        arrays["__capture_info_json"] = np.frombuffer(
            json.dumps(capture_info, sort_keys=True).encode("utf-8"), dtype=np.uint8
        )
        _robust_savez_compressed(path, arrays)
        return dimensions

    return save_capture_npz


base._save_capture_npz_factory = _save_capture_npz_factory_v2


def main() -> None:
    base.main()


if __name__ == "__main__":
    main()
