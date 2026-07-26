from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from .attacks import add_control_targets, load_matrix, read_metadata


def heldout_cluster_diagnostics(
    metadata_paths: Sequence[str | Path],
    group: str,
    observation: str = "observed",
    n_clusters: Optional[int] = None,
    pca_dim: int = 64,
    seed: int = 42,
    output_csv: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Cluster only supplied held-out observations and compare structure to multiple candidate factors.

    ARI/NMI do not require post-hoc semantic cluster naming, so the analysis can
    test whether geometry aligns more with dominant label, modality, client
    identity or communication stage without manually mapping clusters to roles.
    """
    frame = add_control_targets(read_metadata(metadata_paths), random_seed=seed)
    x = load_matrix(frame, group, observation)
    x = StandardScaler().fit_transform(x)
    if pca_dim and x.shape[1] > pca_dim:
        dim = min(int(pca_dim), x.shape[0] - 1, x.shape[1])
        x = PCA(n_components=dim, random_state=seed).fit_transform(x)

    if n_clusters is None:
        n_clusters = max(2, int(frame["dominant_label"].nunique()))
    labels = KMeans(n_clusters=int(n_clusters), random_state=seed, n_init=20).fit_predict(x)
    row = {
        "group": group,
        "observation": observation,
        "n_updates": len(frame),
        "n_clusters": int(n_clusters),
        "silhouette": float(silhouette_score(x, labels)) if len(set(labels)) > 1 else float("nan"),
        "davies_bouldin": float(davies_bouldin_score(x, labels)) if len(set(labels)) > 1 else float("nan"),
        "calinski_harabasz": float(calinski_harabasz_score(x, labels)) if len(set(labels)) > 1 else float("nan"),
    }
    targets = {
        "dominant_label": frame["dominant_label"].to_numpy(),
        "modality": frame["modality_target"].to_numpy(),
        "client_identity": frame["client_identity"].to_numpy(),
        "round_group": frame["round_group"].to_numpy(),
        "random_group": frame["random_group"].to_numpy(),
    }
    for name, target in targets.items():
        row[f"ari_{name}"] = float(adjusted_rand_score(target, labels))
        row[f"nmi_{name}"] = float(normalized_mutual_info_score(target, labels))
    result = pd.DataFrame([row])
    if output_csv:
        path = Path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(path, index=False)
    return result
