from __future__ import annotations

from functools import lru_cache
from typing import Mapping

import numpy as np
import torch

from .defenses import (
    apply_defense_to_state as apply_legacy_defense_to_state,
    apply_vector_to_delta,
    delta_to_state,
    flatten_delta,
    load_basis,
    state_delta,
    transform_vector,
)
from .l2_budget import apply_common_l2_cap_inplace
from .hash_subspace import (
    clip_gaussian_delta_inplace,
    load_hash_basis,
    project_delta_inplace,
)


_HASH_NAMES = {
    "implicit_hash_filter",
    "implicit_hash_contrast",
    "implicit_hash_pca",
    "implicit_hash_random",
}


@lru_cache(maxsize=32)
def _cached_exact_basis(path: str, group: str) -> np.ndarray:
    return load_basis(path, group=group)


@lru_cache(maxsize=32)
def _cached_hash_basis(path: str, group: str) -> np.ndarray:
    return load_hash_basis(path, group=group)


def _apply_hash(delta, cfg: dict, seed: int) -> dict:
    group = str(cfg.get("group", "full_update"))
    basis_path = cfg.get("basis_path")
    if not basis_path:
        raise ValueError(f"{cfg.get('name')} requires basis_path")
    projection_dim = int(cfg.get("projection_dim", 512))
    projection_seed = int(cfg.get("projection_seed", 20260820))
    basis = _cached_hash_basis(str(basis_path), group)
    detail = project_delta_inplace(
        delta,
        group=group,
        basis=basis,
        alpha=float(cfg.get("alpha", 1.0)),
        projection_dim=projection_dim,
        seed=projection_seed,
        rank=int(cfg["rank"]) if cfg.get("rank") is not None else None,
    )
    detail["name"] = str(cfg.get("name"))
    detail["basis_path"] = str(basis_path)
    detail["projection_seed"] = projection_seed
    return detail


def _apply_exact(delta, cfg: dict, seed: int) -> dict:
    group = str(cfg.get("group", "classifier_head"))
    vector, layout = flatten_delta(delta, group=group)
    if vector.size == 0:
        return {"group": group, "skipped": True, "reason": "empty group"}
    before_norm = float(np.linalg.norm(vector))

    # Defense70 fit writes orthonormal columns. Cache and use them directly;
    # re-running a D-by-r QR on the ~1.3M-dimensional fusion basis for every
    # client update would add needless CPU/memory overhead.
    basis_path = cfg.get("basis_path")
    if str(cfg.get("name")) == "contrast_filter" and basis_path:
        basis = _cached_exact_basis(str(basis_path), group)
        if cfg.get("rank") is not None:
            basis = basis[:, : int(cfg["rank"])]
        b = np.asarray(basis, dtype=np.float32)
        projection = b @ (b.T @ vector)
        transformed = (vector - float(cfg.get("alpha", 1.0)) * projection).astype(np.float32)
    else:
        transformed = transform_vector(vector, cfg, seed=seed, group=group)

    apply_vector_to_delta(delta, transformed, layout)
    return {
        "group": group,
        "name": cfg.get("name"),
        "before_norm": before_norm,
        "after_norm": float(np.linalg.norm(transformed)),
        "rank": cfg.get("rank"),
        "alpha": cfg.get("alpha"),
        "basis_path": cfg.get("basis_path"),
    }


def _enforce_shared_l2(before_state, after_state, delta, cfg: dict, details: dict) -> dict:
    if "max_relative_update_distortion_l2" not in cfg:
        raise ValueError(
            "Paper matched/PCA/random/ASLWF defense missing an actual common L2 budget. "
            "Regenerate locked parameters and job configs; do not silently skip."
        )
    audit = apply_common_l2_cap_inplace(
        before_state, after_state, delta,
        max_relative_l2=float(cfg["max_relative_update_distortion_l2"]),
    )
    if not audit["budget_verified"]:
        raise AssertionError("Transmitted update exceeded the prescribed L2 perturbation budget")
    details["l2_budget"] = audit
    return details


def apply_defense_to_state(
    before_state: Mapping[str, torch.Tensor],
    after_state: Mapping[str, torch.Tensor],
    defense_cfg: dict | None,
    seed: int,
):
    """Defense70-aware transform dispatcher.

    The legacy implementation remains unchanged for old experiments. Defense70
    adds implicit hash-lifted low-rank projectors for very large encoder/full
    update spaces and a streaming full-update clipping+Gaussian baseline. All
    transforms operate on the client update before the state is returned to
    FedAvg, so the server aggregates the defended update.
    """
    if not defense_cfg:
        return apply_legacy_defense_to_state(before_state, after_state, defense_cfg, seed)

    name = str(defense_cfg.get("name", "none"))
    if name in _HASH_NAMES:
        delta = state_delta(before_state, after_state)
        detail = _apply_hash(delta, defense_cfg, seed)
        if name in ("implicit_hash_contrast", "implicit_hash_pca", "implicit_hash_random"):
            detail = _enforce_shared_l2(before_state, after_state, delta, defense_cfg, detail)
        return delta_to_state(before_state, delta), {"name": name, **detail}

    if name == "contrast_filter" and defense_cfg.get("basis_path"):
        delta = state_delta(before_state, after_state)
        detail = _apply_exact(delta, defense_cfg, seed)
        detail = _enforce_shared_l2(before_state, after_state, delta, defense_cfg, detail)
        return delta_to_state(before_state, delta), {"name": name, **detail}

    if name == "streaming_clip_gaussian":
        delta = state_delta(before_state, after_state)
        group = str(defense_cfg.get("group", "full_update"))
        detail = clip_gaussian_delta_inplace(
            delta,
            group=group,
            clip_norm=float(defense_cfg["clip_norm"]),
            noise_l2_ratio=float(defense_cfg.get("noise_l2_ratio", 0.0)),
            seed=seed,
        )
        return delta_to_state(before_state, delta), {"name": name, **detail}

    if name == "layerwise_mixed_filter":
        layers = list(defense_cfg.get("layers") or [])
        if not layers:
            raise ValueError("layerwise_mixed_filter requires defense.layers")
        delta = state_delta(before_state, after_state)
        details = []
        for idx, layer in enumerate(layers):
            cfg = dict(layer)
            layer_name = str(cfg.get("name", "contrast_filter"))
            layer_seed = int(seed) + idx * 1009
            if layer_name in _HASH_NAMES:
                details.append(_apply_hash(delta, cfg, layer_seed))
            else:
                details.append(_apply_exact(delta, cfg, layer_seed))
        summary = _enforce_shared_l2(before_state, after_state, delta, defense_cfg, {
            "name": name,
            "group": "layerwise",
            "layers": details,
        })
        return delta_to_state(before_state, delta), summary

    return apply_legacy_defense_to_state(before_state, after_state, defense_cfg, seed)
