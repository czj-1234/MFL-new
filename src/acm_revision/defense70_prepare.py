from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from . import core72_plan
from .data_protocol import (
    create_matched_composition_pair,
    load_json,
    make_client_specs,
    partition_clients_strict,
)
from .defenses import name_in_group
from .fl import build_runtime, local_train, set_global_seed
from .hash_subspace import sketch_state_delta, state_group_dimension
from .orchestrator import load_yaml

REFERENCE_ROUNDS = (1, 10, 30, 50, 75, 100, 125, 150)
EXACT_GROUPS = ("classifier_head", "fusion", "missing_modality")
HASH_GROUPS = ("image_encoder", "text_encoder", "full_update")
PROJECTION_DIM = 512
PROJECTION_SEED = 20260820
MAX_RANK = 5


def _run_id(cfg: dict, population: str) -> str:
    exp = cfg["experiment"]
    fed = cfg["federated"]
    defense = cfg.get("defense", {}).get("name", "none")
    concentration = exp.get("concentration", exp.get("association", "iid"))
    model = cfg.get("model", {}).get("architecture", "clip_dual")
    return (
        f"{cfg['data'].get('name','dataset')}__{model}__{population}__{exp['setting_name']}__c{concentration}"
        f"__n{fed['num_clients']}__p{fed.get('participation_rate',1.0)}__{fed.get('aggregation','fedavg')}"
        f"__seed{cfg['seed']}__def-{defense}"
    ).replace("/", "-")


def make_reference_config(
    base_config: str | Path,
    output_config: str | Path,
    output_root: str | Path = "results/acm_revision/defense70_prep/reference",
) -> dict:
    cfg = copy.deepcopy(load_yaml(base_config))
    cfg["seed"] = 142
    fed = cfg["federated"]
    fed.update(
        {
            "num_clients": 12,
            "partition_mode": "fixed",
            "samples_per_client": 200,
            "rounds": 150,
            "participation_rate": 1.0,
            "min_participants": 12,
            "local_epochs": 1,
            "optimizer": "adamw",
            "lr": 0.000001,
            "weight_decay": 0.05,
            "batch_size": 16,
            "max_local_steps": None,
            "aggregation": "fedavg",
            "fedprox_mu": 0.0,
        }
    )
    cfg.setdefault("evaluation", {})["eval_every"] = 5
    cfg["evaluation"]["num_workers"] = 2
    exp = cfg["experiment"]
    exp.update(
        {
            "population": "shadow_train",
            "setting_name": "modality_exclusive",
            "concentration": 0.7,
            "output_root": str(output_root),
            "job_id": "defense70_shadow_reference_seed142",
            "purpose": "shadow_only_reference_checkpoints_for_defense_basis",
        }
    )
    # The reference run exists only to provide shadow-side global checkpoints.
    # Keep capture intentionally small; formal Defense70 runs do the rich capture.
    cfg["privacy_capture"] = {
        "every_round_exact_groups": ["classifier_head"],
        "milestone_exact_groups": [],
        "milestone_sketch_groups": [],
        "milestone_rounds": list(REFERENCE_ROUNDS),
        "sketch_dim": 512,
        "sketch_seed": PROJECTION_SEED,
    }
    cfg["update_capture"] = {
        "save_all_checkpoints": False,
        "checkpoint_rounds": list(REFERENCE_ROUNDS),
        "model_checkpoint_rounds": list(REFERENCE_ROUNDS),
        "storage_dtype": "float32",
        "groups": ["classifier_head"],
        "retain_for_defense70_shadow_contrast": True,
    }
    cfg["defense"] = {"name": "none", "group": "classifier_head"}

    output_config = Path(output_config)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    with output_config.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return {
        "config": str(output_config),
        "run_dir": str(Path(output_root) / _run_id(cfg, "shadow_train")),
        "checkpoint_rounds": list(REFERENCE_ROUNDS),
    }


def _exact_state_delta_vector(
    before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor], group: str
) -> np.ndarray:
    parts = []
    for name in sorted(before):
        base = before[name]
        if not torch.is_floating_point(base) or not name_in_group(name, group):
            continue
        if name not in after:
            raise KeyError(name)
        diff = after[name].detach().cpu().float() - base.detach().cpu().float()
        parts.append(diff.numpy().reshape(-1).astype(np.float32, copy=False))
    if not parts:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(parts).astype(np.float32, copy=False)


def _state_delta_norm(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]) -> float:
    total = 0.0
    for name in sorted(before):
        base = before[name]
        if not torch.is_floating_point(base) or name not in after:
            continue
        left = base.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        right = after[name].detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        block = right.astype(np.float64, copy=False) - left.astype(np.float64, copy=False)
        total += float(np.dot(block, block))
    return float(np.sqrt(max(0.0, total)))


def _extract_features(before, after) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for group in EXACT_GROUPS:
        out[group] = _exact_state_delta_vector(before, after, group)
    for group in HASH_GROUPS:
        out[group] = sketch_state_delta(
            before,
            after,
            group=group,
            projection_dim=PROJECTION_DIM,
            seed=PROJECTION_SEED,
        )
    return out


def _fit_lowrank(rows: Sequence[np.ndarray], max_rank: int = MAX_RANK) -> tuple[np.ndarray, dict]:
    if not rows:
        raise ValueError("No rows supplied for basis fitting")
    x = np.stack(rows, axis=0).astype(np.float32, copy=False)
    if x.shape[1] <= 0:
        raise ValueError("Cannot fit a basis for an empty update group")
    xc = x - x.mean(axis=0, keepdims=True)
    gram = xc @ xc.T
    values, vectors = np.linalg.eigh(gram.astype(np.float64, copy=False))
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]
    positive = np.where(values > max(1e-12, values[0] * 1e-10 if values.size else 1e-12))[0]
    rank = min(int(max_rank), len(positive), x.shape[1])
    if rank < 1:
        # Degenerate control: keep one deterministic direction if all rows are equal.
        direction = xc[0] if np.linalg.norm(xc[0]) > 0 else x[0]
        norm = float(np.linalg.norm(direction))
        if norm <= 0:
            direction = np.zeros(x.shape[1], dtype=np.float32)
            direction[0] = 1.0
            norm = 1.0
        basis = (direction / norm)[:, None].astype(np.float32)
        return basis, {"stored_rank": 1, "eigenvalues": [0.0], "degenerate": True}
    s = np.sqrt(values[:rank])
    basis = (xc.T @ vectors[:, :rank]) / (s[None, :] + 1e-12)
    q, _ = np.linalg.qr(basis.astype(np.float64, copy=False))
    basis = q[:, :rank].astype(np.float32)
    energy = values / (values.sum() + 1e-12)
    return basis, {
        "stored_rank": int(rank),
        "eigenvalues": values[:rank].tolist(),
        "cumulative_explained_energy": np.cumsum(energy[:rank]).tolist(),
        "n_rows": int(x.shape[0]),
        "dimension": int(x.shape[1]),
    }


def _save_group_bases(path: Path, bases: dict[str, np.ndarray], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v, dtype=np.float32) for k, v in bases.items()})
    with path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def _apply_feature_filter(x: np.ndarray, basis: np.ndarray, rank: int, alpha: float) -> np.ndarray:
    q, _ = np.linalg.qr(np.asarray(basis[:, : int(rank)], dtype=np.float64))
    q = q.astype(np.float32)
    return (x - float(alpha) * (x @ q) @ q.T).astype(np.float32)


def _attack_score(x_train, y_train, x_val, y_val) -> float:
    scaler = StandardScaler(with_mean=True, with_std=True)
    train = scaler.fit_transform(x_train)
    val = scaler.transform(x_val)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=20260820)
    clf.fit(train, y_train)
    pred = clf.predict(val)
    return float(balanced_accuracy_score(y_val, pred))


def _select_operating_points(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    basis: np.ndarray,
) -> tuple[dict, dict, list[dict]]:
    rows: list[dict] = []
    weak_grid = [(r, a) for r in (1, 3) for a in (0.25, 0.5)]
    strong_grid = [(r, a) for r in (3, 5) for a in (0.75, 1.0)]
    max_available = int(basis.shape[1])
    seen = set()
    for tier, grid in (("weak", weak_grid), ("strong", strong_grid)):
        for rank, alpha in grid:
            rank = min(int(rank), max_available)
            key = (tier, rank, float(alpha))
            if key in seen or rank < 1:
                continue
            seen.add(key)
            tx = _apply_feature_filter(train_x, basis, rank, alpha)
            vx = _apply_feature_filter(val_x, basis, rank, alpha)
            score = _attack_score(tx, train_y, vx, val_y)
            rel = np.linalg.norm(vx - val_x, axis=1) / (np.linalg.norm(val_x, axis=1) + 1e-12)
            rows.append(
                {
                    "tier": tier,
                    "rank": rank,
                    "alpha": float(alpha),
                    "shadow_val_attack_balanced_acc": score,
                    "shadow_val_mean_relative_sketch_change": float(np.mean(rel)),
                }
            )
    if not rows:
        raise RuntimeError("No valid rank/alpha candidates")

    def choose(tier: str) -> dict:
        candidates = [r for r in rows if r["tier"] == tier]
        candidates.sort(
            key=lambda r: (
                r["shadow_val_attack_balanced_acc"],
                r["shadow_val_mean_relative_sketch_change"],
                r["rank"],
                r["alpha"],
            )
        )
        return dict(candidates[0])

    weak, strong = choose("weak"), choose("strong")
    return weak, strong, rows


def fit_shadow_only_preparation(
    reference_config: str | Path,
    reference_run_dir: str | Path,
    output_dir: str | Path = "results/acm_revision/defense70_prep",
    samples_per_branch: int = 100,
) -> dict:
    cfg = load_yaml(reference_config)
    reference_run_dir = Path(reference_run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(142)
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))

    pools_dir = Path(cfg["data"]["pools_dir"])
    shadow_train = load_json(pools_dir / "shadow_train.json")
    shadow_val = load_json(pools_dir / "shadow_val.json")
    num_classes = int(cfg["data"]["num_classes"])
    specs = make_client_specs(12, "modality_exclusive", num_classes)

    missing = [
        str(reference_run_dir / "checkpoints" / f"round_{r:04d}.pt")
        for r in REFERENCE_ROUNDS
        if not (reference_run_dir / "checkpoints" / f"round_{r:04d}.pt").exists()
    ]
    if missing:
        raise FileNotFoundError(f"Reference checkpoints missing; first={missing[0]}")

    model, tokenizer, image_processor = build_runtime(cfg)
    contrast_rows = {group: [] for group in (*EXACT_GROUPS, *HASH_GROUPS)}
    concentrated_rows = {group: [] for group in (*EXACT_GROUPS, *HASH_GROUPS)}
    train_labels = []

    print("[DEFENSE70 PREP 2/4] collecting matched shadow-train contrasts", flush=True)
    for round_id in REFERENCE_ROUNDS:
        checkpoint = reference_run_dir / "checkpoints" / f"round_{round_id:04d}.pt"
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.to("cpu")
        before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        for spec in specs:
            pair_seed = 142 * 1_000_000 + int(round_id) * 10_000 + spec.client_id
            conc_samples, bal_samples, _ = create_matched_composition_pair(
                pool=shadow_train,
                spec=spec,
                num_classes=num_classes,
                samples_per_client=int(samples_per_branch),
                concentrated=0.7,
                balanced=0.5,
                seed=pair_seed,
            )
            conc_after, _ = local_train(
                model, conc_samples, tokenizer, image_processor, spec.modality, cfg, device, pair_seed
            )
            conc_feat = _extract_features(before, conc_after)
            del conc_after
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            bal_after, _ = local_train(
                model, bal_samples, tokenizer, image_processor, spec.modality, cfg, device, pair_seed
            )
            bal_feat = _extract_features(before, bal_after)
            del bal_after
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            for group in contrast_rows:
                contrast_rows[group].append((conc_feat[group] - bal_feat[group]).astype(np.float32))
                concentrated_rows[group].append(conc_feat[group].astype(np.float32))
            train_labels.append(int(spec.dominant_label))
        del before, state
        gc.collect()

    contrast_bases: dict[str, np.ndarray] = {}
    contrast_meta = {}
    for group, rows in contrast_rows.items():
        basis, diag = _fit_lowrank(rows, MAX_RANK)
        contrast_bases[group] = basis
        contrast_meta[group] = diag
    contrast_path = output_dir / "bases" / "contrast_bases.npz"
    _save_group_bases(
        contrast_path,
        contrast_bases,
        {
            "method": "matched_label_composition_contrast",
            "fit_population": "shadow_train",
            "target_data_used": False,
            "reference_rounds": list(REFERENCE_ROUNDS),
            "concentrated": 0.7,
            "balanced": 0.5,
            "samples_per_branch": int(samples_per_branch),
            "hash_groups": list(HASH_GROUPS),
            "projection_dim": PROJECTION_DIM,
            "projection_seed": PROJECTION_SEED,
            "diagnostics": contrast_meta,
        },
    )

    # Rank-matched PCA/random controls are fitted/created in exactly the same
    # full-update hash space as the proposed single-full-update basis.
    pca_basis, pca_diag = _fit_lowrank(concentrated_rows["full_update"], MAX_RANK)
    pca_path = output_dir / "bases" / "full_update_pca.npz"
    _save_group_bases(
        pca_path,
        {"full_update": pca_basis},
        {
            "method": "rank_matched_pca",
            "fit_population": "shadow_train",
            "target_data_used": False,
            "projection_dim": PROJECTION_DIM,
            "projection_seed": PROJECTION_SEED,
            "diagnostics": pca_diag,
        },
    )
    rng = np.random.default_rng(20260820)
    random_matrix = rng.normal(size=(PROJECTION_DIM, MAX_RANK))
    random_basis, _ = np.linalg.qr(random_matrix)
    random_path = output_dir / "bases" / "full_update_random.npz"
    _save_group_bases(
        random_path,
        {"full_update": random_basis[:, :MAX_RANK].astype(np.float32)},
        {
            "method": "rank_matched_random_subspace",
            "seed": 20260820,
            "target_data_used": False,
            "projection_dim": PROJECTION_DIM,
            "projection_seed": PROJECTION_SEED,
        },
    )

    print("[DEFENSE70 PREP 3/4] shadow-validation rank/attenuation calibration", flush=True)
    val_clients, val_partition = partition_clients_strict(
        pool=shadow_val,
        specs=specs,
        num_classes=num_classes,
        concentration=0.7,
        samples_per_client=100,
        seed=242,
        partition_mode="fixed",
    )
    val_full_rows = []
    val_labels = []
    val_full_norms = []
    for round_id in REFERENCE_ROUNDS:
        checkpoint = reference_run_dir / "checkpoints" / f"round_{round_id:04d}.pt"
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.to("cpu")
        before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        for spec in specs:
            local_seed = 242 * 1_000_000 + int(round_id) * 10_000 + spec.client_id
            after, _ = local_train(
                model,
                val_clients[spec.client_id],
                tokenizer,
                image_processor,
                spec.modality,
                cfg,
                device,
                local_seed,
            )
            val_full_rows.append(
                sketch_state_delta(
                    before,
                    after,
                    group="full_update",
                    projection_dim=PROJECTION_DIM,
                    seed=PROJECTION_SEED,
                )
            )
            val_full_norms.append(_state_delta_norm(before, after))
            val_labels.append(int(spec.dominant_label))
            del after
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del before, state
        gc.collect()

    train_x = np.stack(concentrated_rows["full_update"]).astype(np.float32)
    train_y = np.asarray(train_labels, dtype=np.int64)
    val_x = np.stack(val_full_rows).astype(np.float32)
    val_y = np.asarray(val_labels, dtype=np.int64)
    weak, strong, tuning_rows = _select_operating_points(
        train_x, train_y, val_x, val_y, contrast_bases["full_update"]
    )

    clip_weak = float(np.quantile(np.asarray(val_full_norms, dtype=np.float64), 0.75))
    clip_strong = float(np.quantile(np.asarray(val_full_norms, dtype=np.float64), 0.50))
    manifest = {
        "protocol": "Defense70-v1",
        "status": "LOCKED",
        "scientific_scope": {
            "dataset": "hateful_memes",
            "setting": "modality_exclusive",
            "concentration": 0.7,
            "rounds": 150,
            "num_clients": 12,
            "participation_rate": 1.0,
            "local_epochs": 1,
            "batch_size": 16,
            "aggregation": "fedavg",
            "optimizer": "adamw",
            "lr": 0.000001,
            "weight_decay": 0.05,
        },
        "provenance": {
            "basis_population": "shadow_train",
            "basis_seed": 142,
            "validation_population": "shadow_val",
            "validation_seed": 242,
            "target_data_used_for_basis_or_tuning": False,
            "reference_run_dir": str(reference_run_dir),
            "reference_rounds": list(REFERENCE_ROUNDS),
            "matched_contrast_samples_per_branch": int(samples_per_branch),
            "shadow_val_partition_unique_assigned": val_partition.get("unique_assigned"),
        },
        "implicit_hash_lift": {
            "projection_dim": PROJECTION_DIM,
            "projection_seed": PROJECTION_SEED,
            "all_selected_group_coordinates_consumed": True,
            "original_space_interpretation": "B = H^T U with row-orthonormal H; B is an implicit low-rank basis in the original update coordinates",
        },
        "basis_paths": {
            "contrast": str(contrast_path),
            "pca_full_update": str(pca_path),
            "random_full_update": str(random_path),
        },
        "selected": {
            "weak": {"rank": int(weak["rank"]), "alpha": float(weak["alpha"])},
            "strong": {"rank": int(strong["rank"]), "alpha": float(strong["alpha"])},
            "head_ablation": {"rank": int(strong["rank"]), "alpha": float(strong["alpha"])},
            "single_full_update_basis": {"rank": int(strong["rank"]), "alpha": float(strong["alpha"])},
            "clip_gaussian_weak": {
                "clip_norm": clip_weak,
                "noise_l2_ratio": 0.05,
            },
            "clip_gaussian_strong": {
                "clip_norm": clip_strong,
                "noise_l2_ratio": 0.15,
            },
        },
        "tuning_budget": {
            "weak_grid": {"ranks": [1, 3], "alphas": [0.25, 0.5]},
            "strong_grid": {"ranks": [3, 5], "alphas": [0.75, 1.0]},
            "selection_metric": "minimum shadow-validation dominant-label balanced accuracy; relative sketch change breaks ties",
            "candidate_results": tuning_rows,
        },
        "clip_calibration": {
            "weak_clip_quantile": 0.75,
            "strong_clip_quantile": 0.50,
            "validation_full_update_norm_count": len(val_full_norms),
        },
        "formal_run_populations": {
            "shadow_train": [142],
            "shadow_val": [242],
            "target": [42, 43, 44, 45, 46],
        },
    }
    manifest_path = output_dir / "locked_params.yaml"
    with manifest_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True)
    with (output_dir / "prep_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "contrast_basis": contrast_meta,
                "pca": pca_diag,
                "tuning": tuning_rows,
                "val_full_update_norms": val_full_norms,
                "target_data_used": False,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("[DEFENSE70 PREP 4/4] parameters locked; target data were not used", flush=True)
    return {"manifest": str(manifest_path), "status": "LOCKED", "selected": manifest["selected"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the shadow-only Defense70 protocol")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("write-reference-config")
    p.add_argument("--base-config", default="configs/acm_revision/hateful_memes.yaml")
    p.add_argument("--output-config", default="configs/acm_revision/generated/defense70_prep/reference_seed142.yaml")
    p.add_argument("--output-root", default="results/acm_revision/defense70_prep/reference")

    p = sub.add_parser("fit")
    p.add_argument("--reference-config", required=True)
    p.add_argument("--reference-run-dir", required=True)
    p.add_argument("--output-dir", default="results/acm_revision/defense70_prep")
    p.add_argument("--samples-per-branch", type=int, default=100)

    args = parser.parse_args()
    if args.command == "write-reference-config":
        print(json.dumps(make_reference_config(args.base_config, args.output_config, args.output_root), indent=2))
        return
    result = fit_shadow_only_preparation(
        args.reference_config,
        args.reference_run_dir,
        args.output_dir,
        args.samples_per_branch,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
