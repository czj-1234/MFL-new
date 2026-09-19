from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .attacks import _load_vector, read_metadata


def build_secure_aggregate_observations(
    metadata_paths: Sequence[str | Path],
    group: str = "full_update",
    observation: str = "observed",
    output_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Replace individual client updates with one weighted aggregate per round.

    Secure aggregation is not an update sanitization baseline: it changes the
    server's observation space. Individual dominant-label targets therefore no
    longer have a one-to-one observation. This helper materializes exactly that
    aggregate observation for a system-level comparison and records the cohort
    label composition instead of pretending an individual-label ASR is defined.
    """
    frame = read_metadata(metadata_paths)
    rows = []
    output_root = Path(output_dir) if output_dir else None
    if output_root:
        output_root.mkdir(parents=True, exist_ok=True)

    group_columns = [c for c in ("run_id", "population", "round") if c in frame.columns]
    for keys, subset in frame.groupby(group_columns, dropna=False):
        weights = subset["num_samples"].to_numpy(dtype=np.float64)
        weights = weights / (weights.sum() + 1e-12)
        vectors = [_load_vector(path, group, observation) for path in subset["update_path"].tolist()]
        aggregate = np.zeros_like(vectors[0], dtype=np.float64)
        for weight, vector in zip(weights, vectors):
            aggregate += float(weight) * vector.astype(np.float64)
        aggregate = aggregate.astype(np.float32)

        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, key_values))
        row.update(
            {
                "group": group,
                "observation_model": "secure_aggregation",
                "num_clients_in_aggregate": int(len(subset)),
                "individual_client_identity_observable": False,
                "individual_dominant_label_target_defined": False,
            }
        )
        dominant_counts = subset["dominant_label"].value_counts(normalize=True)
        for label, proportion in dominant_counts.items():
            row[f"cohort_dominant_label_prop_{int(label)}"] = float(proportion)
        if output_root:
            run_id = str(row.get("run_id", "run")).replace("/", "-")
            round_id = int(row.get("round", 0))
            path = output_root / f"{run_id}__round_{round_id:04d}__{group}.npz"
            np.savez_compressed(path, aggregate=aggregate)
            row["aggregate_path"] = str(path)
        rows.append(row)

    result = pd.DataFrame(rows)
    if output_root:
        result.to_csv(output_root / "secure_aggregation_metadata.csv", index=False)
        with (output_root / "README.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "interpretation": "system-level observation-model alternative",
                    "warning": "Do not report individual dominant-label ASR because the server no longer observes individual client updates.",
                },
                f,
                indent=2,
            )
    return result
