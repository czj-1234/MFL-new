from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from .data_protocol import create_matched_composition_pair, load_json, make_client_specs
from .defenses import flatten_delta, state_delta
from .fl import build_runtime, local_train, set_global_seed


def _checkpoint_path(reference_run_dir: str | Path, round_id: int) -> Path:
    return Path(reference_run_dir) / "checkpoints" / f"round_{round_id:04d}.pt"


def collect_identical_checkpoint_contrasts(
    cfg: dict,
    reference_run_dir: str | Path,
    pool_json: str | Path,
    output_dir: str | Path,
    checkpoint_rounds: Sequence[int],
    groups: Sequence[str],
    concentrated: float,
    balanced: Optional[float] = None,
    clients: Optional[Sequence[int]] = None,
    samples_per_branch: Optional[int] = None,
) -> dict:
    """Collect matched-run update contrasts from exactly the same global checkpoints.

    The global model state, client identity, modality, optimizer configuration,
    local step budget and shuffle seed are held fixed for each pair. The local
    label composition is changed between concentrated and balanced branches.
    This supports an associative label-composition interpretation without the
    divergent-global-trajectory confound in the original design.
    """
    seed = int(cfg["seed"])
    set_global_seed(seed)
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    pool = load_json(pool_json)
    num_classes = int(cfg["data"]["num_classes"])
    setting = cfg["experiment"]["setting_name"]
    specs = make_client_specs(int(cfg["federated"]["num_clients"]), setting, num_classes)
    if clients is not None:
        client_set = set(int(x) for x in clients)
        specs = [s for s in specs if s.client_id in client_set]
    if samples_per_branch is None:
        samples_per_branch = int(cfg["federated"].get("samples_per_client") or max(1, len(pool) // max(1, len(specs))))
    if balanced is None:
        balanced = 1.0 / num_classes

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, image_processor = build_runtime(cfg)
    records: List[dict] = []
    group_vectors: Dict[str, List[np.ndarray]] = {group: [] for group in groups}

    for round_id in checkpoint_rounds:
        checkpoint = _checkpoint_path(reference_run_dir, int(round_id))
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Missing {checkpoint}. The reference FL run must use update_capture.save_all_checkpoints=true."
            )
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.to("cpu")
        before_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        for spec in specs:
            pair_seed = seed * 1_000_000 + int(round_id) * 10_000 + spec.client_id
            conc_samples, bal_samples, match_report = create_matched_composition_pair(
                pool=pool,
                spec=spec,
                num_classes=num_classes,
                samples_per_client=samples_per_branch,
                concentrated=float(concentrated),
                balanced=float(balanced),
                seed=pair_seed,
            )
            conc_after, conc_metrics = local_train(
                model,
                conc_samples,
                tokenizer,
                image_processor,
                spec.modality,
                cfg,
                device,
                pair_seed,
            )
            # Same global checkpoint and same DataLoader seed for the matched branch.
            bal_after, bal_metrics = local_train(
                model,
                bal_samples,
                tokenizer,
                image_processor,
                spec.modality,
                cfg,
                device,
                pair_seed,
            )
            conc_delta = state_delta(before_state, conc_after)
            bal_delta = state_delta(before_state, bal_after)
            contrast_delta = {name: conc_delta[name] - bal_delta[name] for name in conc_delta if name in bal_delta}
            record_path = output_dir / "contrasts" / f"round_{int(round_id):04d}_client_{spec.client_id:04d}.npz"
            record_path.parent.mkdir(parents=True, exist_ok=True)
            arrays = {}
            dims = {}
            for group in groups:
                vector, _ = flatten_delta(contrast_delta, group)
                arrays[group] = vector.astype(np.float32)
                group_vectors[group].append(vector.astype(np.float32))
                dims[group] = int(vector.size)
            np.savez_compressed(record_path, **arrays)
            records.append(
                {
                    "round": int(round_id),
                    "client_id": spec.client_id,
                    "modality": spec.modality,
                    "dominant_label": spec.dominant_label,
                    "contrast_path": str(record_path),
                    "concentrated": float(concentrated),
                    "balanced": float(balanced),
                    "shared_example_fraction": match_report["shared_fraction"],
                    "conc_local_loss": conc_metrics["local_loss"],
                    "bal_local_loss": bal_metrics["local_loss"],
                    "dimensions": dims,
                    "matched_variables": [
                        "global_checkpoint",
                        "client_identity",
                        "modality",
                        "sample_budget",
                        "optimizer_configuration",
                        "learning_rate",
                        "local_epochs",
                        "local_step_budget",
                        "shuffle_seed",
                    ],
                    "intentionally_changed_variable": "local_label_composition",
                    "sample_instances_fully_identical": match_report["shared_fraction"] == 1.0,
                }
            )

    with (output_dir / "contrast_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return {
        "records": records,
        "vectors": group_vectors,
        "interpretation": "identical-checkpoint matched label-composition contrast",
    }


def fit_contrast_bases(
    contrast_result: dict,
    output_path: str | Path,
    max_rank: Optional[int] = None,
) -> dict:
    """Fit an SVD basis for every requested observation group.

    The full ordered basis is saved; rank must be selected later on an
    independent shadow-validation run, not on target results.
    """
    bases = {}
    diagnostics = {}
    for group, vectors in contrast_result["vectors"].items():
        matrix = np.stack(vectors, axis=0).astype(np.float64)
        matrix = matrix - matrix.mean(axis=0, keepdims=True)
        _, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
        rank = len(singular_values) if max_rank is None else min(int(max_rank), len(singular_values))
        basis = vt[:rank].T.astype(np.float32)
        bases[group] = basis
        energy = singular_values ** 2
        diagnostics[group] = {
            "num_contrasts": int(matrix.shape[0]),
            "dimension": int(matrix.shape[1]),
            "stored_rank": int(rank),
            "singular_values": singular_values[:rank].tolist(),
            "explained_energy": (energy[:rank] / (energy.sum() + 1e-12)).tolist(),
            "cumulative_explained_energy": np.cumsum(energy[:rank] / (energy.sum() + 1e-12)).tolist(),
        }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **bases)
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "interpretation": "label-composition-associated subspace estimated from identical checkpoints",
                "rank_selection_rule": "select on independent shadow-validation data only",
                "diagnostics": diagnostics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return diagnostics


def basis_stability(basis_paths: Sequence[str | Path], group: str, rank: int) -> List[dict]:
    """Pairwise principal-angle diagnostics across seeds/partitions/datasets."""
    loaded = []
    for path in basis_paths:
        with np.load(path, allow_pickle=False) as data:
            basis = np.asarray(data[group], dtype=np.float64)[:, :rank]
        q, _ = np.linalg.qr(basis)
        loaded.append((str(path), q))

    rows = []
    for i in range(len(loaded)):
        for j in range(i + 1, len(loaded)):
            left_name, left = loaded[i]
            right_name, right = loaded[j]
            if left.shape[0] != right.shape[0]:
                rows.append(
                    {
                        "left": left_name,
                        "right": right_name,
                        "group": group,
                        "rank": rank,
                        "compatible": False,
                    }
                )
                continue
            singular = np.linalg.svd(left.T @ right, compute_uv=False)
            singular = np.clip(singular, -1.0, 1.0)
            angles = np.arccos(singular)
            rows.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "group": group,
                    "rank": rank,
                    "compatible": True,
                    "mean_principal_angle_rad": float(np.mean(angles)),
                    "max_principal_angle_rad": float(np.max(angles)),
                    "mean_subspace_cosine": float(np.mean(singular)),
                }
            )
    return rows
