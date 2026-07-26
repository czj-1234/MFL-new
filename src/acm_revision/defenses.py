from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch


LAYER_GROUPS = (
    "classifier_bias",
    "classifier_weight",
    "classifier_head",
    "fusion",
    "image_encoder",
    "text_encoder",
    "missing_modality",
    "all_shared",
    "full_update",
)


def name_in_group(name: str, group: str) -> bool:
    if group == "classifier_bias":
        return name.endswith("classifier.bias") or name.endswith("fusion.classifier.bias")
    if group == "classifier_weight":
        return name.endswith("classifier.weight") or name.endswith("fusion.classifier.weight")
    if group == "classifier_head":
        return "classifier" in name and "image_classifier" not in name and "text_classifier" not in name
    if group == "fusion":
        return "multi_modal_projector" in name or "fusion.multi_modal_projector" in name
    if group == "image_encoder":
        return any(
            token in name
            for token in (
                "clip.vision_model",
                "clip.visual_projection",
                "image_encoder",
                "image_proj",
                "image_norm",
                "image_classifier",
            )
        )
    if group == "text_encoder":
        return any(
            token in name
            for token in (
                "clip.text_model",
                "clip.text_projection",
                "text_encoder",
                "text_proj",
                "text_norm",
                "text_classifier",
            )
        )
    if group == "missing_modality":
        return "missing_image_embedding" in name or "missing_text_embedding" in name
    if group == "all_shared":
        # Shared learnable parameters excluding branch-private output heads.
        return "image_classifier" not in name and "text_classifier" not in name
    if group == "full_update":
        return True
    raise ValueError(f"Unknown layer group: {group}")


def state_delta(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for name, tensor in before.items():
        if name not in after:
            continue
        if not torch.is_floating_point(tensor):
            continue
        out[name] = after[name].detach().cpu().float() - tensor.detach().cpu().float()
    return out


def flatten_delta(
    delta: Mapping[str, torch.Tensor],
    group: str = "full_update",
) -> Tuple[np.ndarray, List[dict]]:
    vectors: List[np.ndarray] = []
    layout: List[dict] = []
    offset = 0
    for name in sorted(delta):
        tensor = delta[name]
        if not name_in_group(name, group):
            continue
        flat = tensor.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        vectors.append(flat)
        layout.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "start": offset,
                "end": offset + flat.size,
            }
        )
        offset += flat.size
    if not vectors:
        return np.zeros((0,), dtype=np.float32), layout
    return np.concatenate(vectors).astype(np.float32, copy=False), layout


def apply_vector_to_delta(
    delta: MutableMapping[str, torch.Tensor],
    vector: np.ndarray,
    layout: Sequence[dict],
) -> None:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    expected = layout[-1]["end"] if layout else 0
    if vector.size != expected:
        raise ValueError(f"Vector length {vector.size} != layout length {expected}.")
    for item in layout:
        segment = vector[item["start"] : item["end"]]
        delta[item["name"]] = torch.from_numpy(segment.reshape(item["shape"]).copy())


def delta_to_state(
    before: Mapping[str, torch.Tensor],
    delta: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for name, tensor in before.items():
        base = tensor.detach().cpu().clone()
        if name in delta and torch.is_floating_point(base):
            out[name] = (base.float() + delta[name].float()).to(dtype=base.dtype)
        else:
            out[name] = base
    return out


def project_out(vector: np.ndarray, basis: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32).reshape(-1)
    b = np.asarray(basis, dtype=np.float32)
    if b.ndim != 2:
        raise ValueError("Basis must be a 2D [dimension, rank] array.")
    if b.shape[0] != v.size:
        raise ValueError(f"Basis dimension {b.shape[0]} does not match update dimension {v.size}.")
    # QR makes the operation robust when the stored basis is not perfectly orthonormal.
    q, _ = np.linalg.qr(b)
    projection = q @ (q.T @ v)
    return (v - float(alpha) * projection).astype(np.float32)


def clip_l2(vector: np.ndarray, clip_norm: float) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32).copy()
    norm = float(np.linalg.norm(v))
    if clip_norm <= 0:
        raise ValueError("clip_norm must be positive.")
    if norm > clip_norm:
        v *= clip_norm / (norm + 1e-12)
    return v


def add_gaussian(vector: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    if std < 0:
        raise ValueError("Gaussian std cannot be negative.")
    return np.asarray(vector, dtype=np.float32) + rng.normal(0.0, std, size=np.asarray(vector).shape).astype(np.float32)


def symmetric_quantize(vector: np.ndarray, bits: int) -> np.ndarray:
    if bits < 2 or bits > 16:
        raise ValueError("bits must be between 2 and 16.")
    v = np.asarray(vector, dtype=np.float32)
    vmax = float(np.max(np.abs(v))) if v.size else 0.0
    if vmax == 0.0:
        return v.copy()
    qmax = (2 ** (bits - 1)) - 1
    scale = vmax / qmax
    return (np.round(v / scale).clip(-qmax, qmax) * scale).astype(np.float32)


def topk(vector: np.ndarray, keep_fraction: float) -> np.ndarray:
    if not 0 < keep_fraction <= 1:
        raise ValueError("keep_fraction must be in (0, 1].")
    v = np.asarray(vector, dtype=np.float32)
    if keep_fraction >= 1 or v.size == 0:
        return v.copy()
    k = max(1, int(math.ceil(v.size * keep_fraction)))
    idx = np.argpartition(np.abs(v), -k)[-k:]
    out = np.zeros_like(v)
    out[idx] = v[idx]
    return out


def random_sparsify(vector: np.ndarray, keep_fraction: float, rng: np.random.Generator) -> np.ndarray:
    if not 0 < keep_fraction <= 1:
        raise ValueError("keep_fraction must be in (0, 1].")
    v = np.asarray(vector, dtype=np.float32)
    mask = rng.random(v.shape) < keep_fraction
    # Unbiased rescaling is useful when the vector is aggregated later.
    return (v * mask.astype(np.float32) / keep_fraction).astype(np.float32)


def random_projection_reconstruct(vector: np.ndarray, rank: int, rng: np.random.Generator) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32)
    if not 1 <= rank <= v.size:
        raise ValueError(f"rank must be in [1, {v.size}].")
    gaussian = rng.normal(size=(v.size, rank)).astype(np.float32)
    q, _ = np.linalg.qr(gaussian)
    return (q @ (q.T @ v)).astype(np.float32)


def load_basis(path: str | Path, group: Optional[str] = None) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    if group and group in loaded.files:
        return np.asarray(loaded[group], dtype=np.float32)
    if "basis" in loaded.files:
        return np.asarray(loaded["basis"], dtype=np.float32)
    if len(loaded.files) == 1:
        return np.asarray(loaded[loaded.files[0]], dtype=np.float32)
    raise KeyError(f"Could not determine basis in {path}; keys={loaded.files}.")


def _basis_for_cfg(cfg: dict, vector_dim: int, group: str, rng: np.random.Generator) -> np.ndarray:
    basis_path = cfg.get("basis_path")
    if basis_path:
        basis = load_basis(basis_path, group=group)
        rank = cfg.get("rank")
        if rank is not None:
            basis = basis[:, : int(rank)]
        return basis
    rank = int(cfg.get("rank", min(1, vector_dim)))
    gaussian = rng.normal(size=(vector_dim, rank)).astype(np.float32)
    q, _ = np.linalg.qr(gaussian)
    return q.astype(np.float32)


def transform_vector(vector: np.ndarray, defense_cfg: Optional[dict], seed: int, group: str) -> np.ndarray:
    if not defense_cfg:
        return np.asarray(vector, dtype=np.float32).copy()
    name = str(defense_cfg.get("name", "none"))
    v = np.asarray(vector, dtype=np.float32).copy()
    rng = np.random.default_rng(seed)

    if name in ("none", "no_defense"):
        return v
    if name == "clip":
        return clip_l2(v, float(defense_cfg["clip_norm"]))
    if name == "gaussian_noise":
        return add_gaussian(v, float(defense_cfg["std"]), rng)
    if name in ("clip_gaussian", "local_dp"):
        clipped = clip_l2(v, float(defense_cfg["clip_norm"]))
        std = float(defense_cfg.get("std", float(defense_cfg.get("noise_multiplier", 1.0)) * float(defense_cfg["clip_norm"])))
        return add_gaussian(clipped, std, rng)
    if name == "quantization":
        return symmetric_quantize(v, int(defense_cfg.get("bits", 8)))
    if name == "topk":
        return topk(v, float(defense_cfg.get("keep_fraction", 0.1)))
    if name == "sparsification":
        return random_sparsify(v, float(defense_cfg.get("keep_fraction", 0.1)), rng)
    if name == "random_projection":
        return random_projection_reconstruct(v, int(defense_cfg["rank"]), rng)
    if name in (
        "random_subspace",
        "rank_matched_random_subspace",
        "pca",
        "rank_matched_pca",
        "contrast_filter",
        "supervised_sensitive_direction",
        "gradient_orthogonalization",
    ):
        basis = _basis_for_cfg(defense_cfg, v.size, group, rng)
        return project_out(v, basis, alpha=float(defense_cfg.get("alpha", 1.0)))
    if name == "secure_aggregation":
        raise ValueError(
            "secure_aggregation changes the server observation model and is not a per-client update transform. "
            "Use observation_model=secure_aggregation in attack evaluation instead."
        )
    raise ValueError(f"Unknown defense: {name}")


def apply_defense_to_state(
    before_state: Mapping[str, torch.Tensor],
    after_state: Mapping[str, torch.Tensor],
    defense_cfg: Optional[dict],
    seed: int,
) -> Tuple[Dict[str, torch.Tensor], dict]:
    """Apply a defense before the client update is exposed/aggregated."""
    if not defense_cfg or defense_cfg.get("name", "none") in ("none", "no_defense"):
        return {k: v.detach().cpu().clone() for k, v in after_state.items()}, {"name": "none"}

    group = defense_cfg.get("group", "full_update")
    delta = state_delta(before_state, after_state)
    vector, layout = flatten_delta(delta, group=group)
    before_norm = float(np.linalg.norm(vector))
    transformed = transform_vector(vector, defense_cfg, seed=seed, group=group)
    apply_vector_to_delta(delta, transformed, layout)
    defended_state = delta_to_state(before_state, delta)
    meta = {
        "name": defense_cfg.get("name"),
        "group": group,
        "before_norm": before_norm,
        "after_norm": float(np.linalg.norm(transformed)),
        "alpha": defense_cfg.get("alpha"),
        "rank": defense_cfg.get("rank"),
        "basis_path": defense_cfg.get("basis_path"),
    }
    return defended_state, meta


def save_basis(path: str | Path, basis: np.ndarray, **metadata) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, basis=np.asarray(basis, dtype=np.float32))
    if metadata:
        with path.with_suffix(".json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
