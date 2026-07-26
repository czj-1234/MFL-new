from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .attacks import AttackSplit, run_tabular_attack


def _subsample_clients(frame: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0,1].")
    clients = frame[["client_key", "dominant_label"]].drop_duplicates()
    rng = np.random.default_rng(seed)
    selected = []
    for _, group in clients.groupby("dominant_label"):
        keys = group["client_key"].astype(str).to_numpy()
        n = max(1, int(round(len(keys) * fraction)))
        chosen = rng.choice(keys, size=min(n, len(keys)), replace=False)
        selected.extend(chosen.tolist())
    return frame[frame["client_key"].astype(str).isin(selected)].copy()


def run_auxiliary_data_sensitivity(
    split: AttackSplit,
    group: str,
    attack_model: str = "mlp",
    target: str = "dominant_label",
    observation: str = "observed",
    fractions: Sequence[float] = (0.10, 0.25, 0.50, 1.00),
    seed: int = 42,
    pca_dim: int = 256,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Vary the amount of auxiliary shadow data by whole client trajectories.

    Validation and target populations remain fixed. Subsampling by whole clients
    avoids creating a misleading gain simply by taking temporally correlated
    client-round observations as independent auxiliary examples.
    """
    rows = []
    for fraction in fractions:
        train = _subsample_clients(split.train, float(fraction), seed)
        reduced = AttackSplit(train=train, val=split.val, test=split.test, protocol=split.protocol)
        result = run_tabular_attack(
            reduced,
            group=group,
            attack_model=attack_model,
            target=target,
            observation=observation,
            seed=seed,
            pca_dim=pca_dim,
        )
        result["shadow_auxiliary_fraction"] = float(fraction)
        result["shadow_auxiliary_clients"] = int(train["client_key"].nunique())
        rows.append(result)
    frame = pd.DataFrame(rows)
    if output_csv:
        path = Path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    return frame
