from __future__ import annotations

import copy
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from .orchestrator import load_yaml, save_yaml


def generate_basis_baseline_sweep(
    base_config: str | Path,
    basis_files: Mapping[str, str | Path],
    output_dir: str | Path,
    group: str = "classifier_head",
    seeds: Sequence[int] = tuple(range(42, 52)),
    populations: Sequence[str] = ("shadow_val", "target"),
    ranks: Sequence[int] = (1, 2, 4, 8, 16),
    alphas: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
) -> pd.DataFrame:
    """Generate exactly rank-matched PCA/random/supervised/orthogonal baselines.

    `basis_files` keys may be `rank_matched_pca`,
    `rank_matched_random_subspace`, `supervised_sensitive_direction`, and
    `gradient_orthogonalization`. Each basis file should be fitted on
    shadow_train at the largest requested rank; lower-rank jobs use its prefix.
    """
    base = load_yaml(base_config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    idx = 0
    for method, basis_path in basis_files.items():
        for seed in seeds:
            for population in populations:
                for rank in ranks:
                    for alpha in alphas:
                        cfg = copy.deepcopy(base)
                        cfg["seed"] = int(seed)
                        cfg["experiment"]["population"] = population
                        cfg["federated"]["samples_per_client"] = 50 if population == "shadow_val" else 100
                        cfg["defense"] = {
                            "name": str(method),
                            "group": group,
                            "basis_path": str(basis_path),
                            "rank": int(rank),
                            "alpha": float(alpha),
                        }
                        job_id = f"basis_baseline_{idx:06d}"
                        cfg["experiment"]["job_id"] = job_id
                        cfg["experiment"]["output_root"] = str(Path("results/acm_revision/defense_sweeps") / output_dir.name / job_id)
                        path = output_dir / f"{job_id}.yaml"
                        save_yaml(cfg, path)
                        rows.append(
                            {
                                "job_id": job_id,
                                "config_path": str(path),
                                "method": method,
                                "basis_path": str(basis_path),
                                "group": group,
                                "seed": seed,
                                "population": population,
                                "rank": rank,
                                "alpha": alpha,
                            }
                        )
                        idx += 1
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "jobs.csv", index=False)
    return frame
