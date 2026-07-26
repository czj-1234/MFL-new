from __future__ import annotations

import copy
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

from .orchestrator import load_yaml, save_yaml


def generate_contrast_defense_sweep(
    base_config: str | Path,
    basis_path: str | Path,
    output_dir: str | Path,
    seeds: Sequence[int] = tuple(range(42, 52)),
    populations: Sequence[str] = ("shadow_val", "target"),
    ranks: Sequence[int] = (1, 2, 4, 8, 16),
    alphas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    modes: Sequence[str] = ("classifier_head", "all_shared", "full_update"),
) -> pd.DataFrame:
    """Generate the complete privacy-utility sweep for the proposed filter."""
    base = load_yaml(base_config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    idx = 0
    for seed in seeds:
        for population in populations:
            for group in modes:
                for rank in ranks:
                    for alpha in alphas:
                        cfg = copy.deepcopy(base)
                        cfg["seed"] = int(seed)
                        cfg["experiment"]["population"] = population
                        cfg["federated"]["samples_per_client"] = 50 if population == "shadow_val" else 100
                        cfg["defense"] = {
                            "name": "contrast_filter",
                            "group": group,
                            "basis_path": str(basis_path),
                            "rank": int(rank),
                            "alpha": float(alpha),
                        }
                        job_id = f"contrast_{idx:06d}"
                        cfg["experiment"]["job_id"] = job_id
                        path = output_dir / f"{job_id}.yaml"
                        save_yaml(cfg, path)
                        rows.append(
                            {
                                "job_id": job_id,
                                "config_path": str(path),
                                "seed": seed,
                                "population": population,
                                "group": group,
                                "rank": rank,
                                "alpha": alpha,
                            }
                        )
                        idx += 1
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "jobs.csv", index=False)
    return result


def generate_layerwise_adaptive_sweep(
    base_config: str | Path,
    basis_path: str | Path,
    output_dir: str | Path,
    seeds: Sequence[int] = tuple(range(42, 52)),
    populations: Sequence[str] = ("shadow_val", "target"),
    ranks: Sequence[int] = (2, 4, 8),
    head_alphas: Sequence[float] = (0.5, 0.75, 1.0),
    fusion_alphas: Sequence[float] = (0.25, 0.5, 0.75),
    encoder_alphas: Sequence[float] = (0.0, 0.25, 0.5),
) -> pd.DataFrame:
    """Generate separate-basis/separate-alpha layer-wise filters.

    The same NPZ may contain group-specific bases created by fit-basis. Each
    layer loads its own group key from that file.
    """
    base = load_yaml(base_config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    idx = 0
    for seed in seeds:
        for population in populations:
            for rank in ranks:
                for head_alpha in head_alphas:
                    for fusion_alpha in fusion_alphas:
                        for encoder_alpha in encoder_alphas:
                            cfg = copy.deepcopy(base)
                            cfg["seed"] = int(seed)
                            cfg["experiment"]["population"] = population
                            cfg["federated"]["samples_per_client"] = 50 if population == "shadow_val" else 100
                            layers = [
                                {
                                    "name": "contrast_filter",
                                    "group": "classifier_head",
                                    "basis_path": str(basis_path),
                                    "rank": int(rank),
                                    "alpha": float(head_alpha),
                                },
                                {
                                    "name": "contrast_filter",
                                    "group": "fusion",
                                    "basis_path": str(basis_path),
                                    "rank": int(rank),
                                    "alpha": float(fusion_alpha),
                                },
                                {
                                    "name": "contrast_filter",
                                    "group": "image_encoder",
                                    "basis_path": str(basis_path),
                                    "rank": int(rank),
                                    "alpha": float(encoder_alpha),
                                },
                                {
                                    "name": "contrast_filter",
                                    "group": "text_encoder",
                                    "basis_path": str(basis_path),
                                    "rank": int(rank),
                                    "alpha": float(encoder_alpha),
                                },
                            ]
                            cfg["defense"] = {"name": "layerwise_contrast_filter", "layers": layers}
                            job_id = f"layerwise_{idx:06d}"
                            cfg["experiment"]["job_id"] = job_id
                            path = output_dir / f"{job_id}.yaml"
                            save_yaml(cfg, path)
                            rows.append(
                                {
                                    "job_id": job_id,
                                    "config_path": str(path),
                                    "seed": seed,
                                    "population": population,
                                    "rank": rank,
                                    "head_alpha": head_alpha,
                                    "fusion_alpha": fusion_alpha,
                                    "encoder_alpha": encoder_alpha,
                                }
                            )
                            idx += 1
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "jobs.csv", index=False)
    return result
