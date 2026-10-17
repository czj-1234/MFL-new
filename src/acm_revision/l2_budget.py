"""Strict common relative-L2 perturbation budget for client updates.

Scientific definition:
    ||Delta_transmitted - Delta_raw||_2 <= rho * ||Delta_raw||_2
where both deltas comprise all floating update coordinates in their original
parameter space (never sketch-space norms). rho is fixed from shadow validation
and shared by matched-contrast, PCA and random methods at a given tier.

This budget acts BEFORE server aggregation. It does NOT claim realized L2 changes
are identical across methods. It is distinct from clip+Gaussian's own clip norm.
"""
from __future__ import annotations

import math
from typing import Mapping

def scaling_factor(raw_norm: float, distortion_norm: float, max_relative_l2: float) -> float:
    """Return scaling to enforce a shared L2 cap without changing direction."""
    r = float(max_relative_l2)
    if not (math.isfinite(r) and 0 < r <= 1.0):
        raise ValueError("max_relative_update_distortion_l2 must be finite in (0, 1]")
    n = float(raw_norm)
    d = float(distortion_norm)
    if not all(math.isfinite(z) and z >= 0 for z in (n, d)):
        raise ValueError("Non-finite/negative raw or distortion L2 norm")
    budget = r * n
    if d == 0:
        return 1.0
    return min(1.0, budget / d)


def apply_common_l2_cap_inplace(
    before_state: Mapping,
    after_state: Mapping,
    filtered_delta: Mapping,
    max_relative_l2: float,
    chunk_size: int = 1_048_576,
) -> dict:
    """Cap actual transmitted-update distortion, including all floating parameters.

    Uses chunked CPU buffers to avoid a second full flattened CLIP update.
    filtered_delta is modified IN PLACE. Nothing uses target labels, and budget
    is fixed in the originating shadow-only config before formal FL training.
    """
    import torch

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    raw_sq, distortion_sq, dimension = 0.0, 0.0, 0
    for name in sorted(filtered_delta):
        tensor = filtered_delta[name]
        if not torch.is_floating_point(tensor):
            continue
        if name not in before_state or name not in after_state:
            raise KeyError("Missing raw parameter coordinates: " + name)
        original = before_state[name].detach().reshape(-1)
        locally_trained = after_state[name].detach().reshape(-1)
        filtered = tensor.detach().reshape(-1)
        if original.numel() != filtered.numel() or locally_trained.numel() != filtered.numel():
            raise ValueError("Inconsistent parameter shape: " + name)
        dimension += int(filtered.numel())
        for offset in range(0, filtered.numel(), chunk_size):
            a = original[offset:offset + chunk_size].to(device="cpu", dtype=torch.float64)
            b = locally_trained[offset:offset + chunk_size].to(device="cpu", dtype=torch.float64)
            c = filtered[offset:offset + chunk_size].to(device="cpu", dtype=torch.float64)
            raw = b - a
            diff = c - raw
            raw_sq += float(torch.dot(raw, raw).item())
            distortion_sq += float(torch.dot(diff, diff).item())
    raw_norm = math.sqrt(max(raw_sq, 0.0))
    proposed_distortion = math.sqrt(max(distortion_sq, 0.0))
    scale = scaling_factor(raw_norm, proposed_distortion, max_relative_l2)
    applied_sq = distortion_sq if scale == 1.0 else 0.0
    if scale < 1.0:
        for name in sorted(filtered_delta):
            tensor = filtered_delta[name]
            if not torch.is_floating_point(tensor):
                continue
            flat = tensor.reshape(-1)
            original = before_state[name].detach().reshape(-1)
            locally_trained = after_state[name].detach().reshape(-1)
            for offset in range(0, flat.numel(), chunk_size):
                sl = slice(offset, offset + chunk_size)
                # Compute in the original dtype to preserve the runner's tensor layout.
                raw = (locally_trained[sl].to(device=flat.device, dtype=flat.dtype) -
                       original[sl].to(device=flat.device, dtype=flat.dtype))
                flat[sl].copy_(raw + scale * (flat[sl] - raw))
                # Verify actual post-rounding original-coordinate perturbation.
                real_change = (flat[sl].detach().to(device="cpu", dtype=torch.float64) -
                               (locally_trained[sl].detach().to(device="cpu", dtype=torch.float64) -
                                original[sl].detach().to(device="cpu", dtype=torch.float64)))
                applied_sq += float(torch.dot(real_change, real_change).item())
    final_distortion = math.sqrt(max(applied_sq, 0.0))
    within_budget = final_distortion <= raw_norm * float(max_relative_l2) * (1.0 + 1e-6) + 1e-7
    if not within_budget:
        raise AssertionError("Actual transmitted update exceeds its original-coordinate L2 cap")
    return {
        "mechanism": "common_relative_raw_update_L2_cap",
        "max_relative_update_distortion_l2": float(max_relative_l2),
        "raw_update_l2": raw_norm,
        "proposed_change_l2": proposed_distortion,
        "allowed_change_l2": raw_norm * float(max_relative_l2),
        "applied_change_l2": final_distortion,
        "applied_scale": scale,
        "num_original_floating_coordinates": dimension,
        "is_active": bool(scale < 1.0),
        "budget_verified": within_budget,
        "scope": "all_transmitted_floating_parameters_before_FedAvg",
    }
