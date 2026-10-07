from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .data_protocol import (
    create_matched_composition_pair,
    load_json,
    make_client_specs,
    partition_clients_strict,
)
from .defenses import flatten_delta, state_delta
from .fl import build_runtime, local_train, set_global_seed
from .orchestrator import load_yaml


REFERENCE_ROUNDS = (1, 10, 30, 50, 75, 100, 125, 150)
EARLY_MAX_ROUND = 60
MAX_RANK = 20
CANDIDATE_RANKS = (1, 3, 5, 10, 20)
CANDIDATE_ALPHAS = (0.5, 0.75, 1.0)


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


def _exact_head_vector(before, after) -> np.ndarray:
    delta = state_delta(before, after)
    vector, _ = flatten_delta(delta, "classifier_head")
    return np.asarray(vector, dtype=np.float32)


def _fit_signed_basis(
    contrasts: list[np.ndarray],
    labels: list[int],
    max_rank: int = MAX_RANK,
) -> tuple[np.ndarray, dict]:
    """Fit a dominant-label-aware basis from matched concentrated-balanced contrasts.

    The first direction is the signed mean contrast:
        q1 proportional to E[(2s-1) * (Delta_c - Delta_bal)].
    Remaining directions are principal residual directions after removing q1.
    """
    x = np.stack(contrasts).astype(np.float32, copy=False)
    y = np.asarray(labels, dtype=np.int64)
    signs = (2 * y - 1).astype(np.float32)
    xs = x * signs[:, None]

    mean_direction = xs.mean(axis=0).astype(np.float64)
    mean_norm = float(np.linalg.norm(mean_direction))
    if mean_norm <= 1e-12:
        # Deterministic fallback for a degenerate signed mean.
        _, _, vt = np.linalg.svd(xs.astype(np.float64), full_matrices=False)
        q1 = vt[0]
        mean_norm = 0.0
    else:
        q1 = mean_direction / mean_norm

    residual = xs.astype(np.float64) - np.outer(xs.astype(np.float64) @ q1, q1)
    residual = residual - residual.mean(axis=0, keepdims=True)

    extra_rank = max(0, min(int(max_rank) - 1, residual.shape[0] - 1, residual.shape[1] - 1))
    if extra_rank > 0:
        gram = residual @ residual.T
        values, left = np.linalg.eigh(gram)
        order = np.argsort(values)[::-1]
        values = np.maximum(values[order], 0.0)
        left = left[:, order]
        positive = np.where(values > max(1e-12, values[0] * 1e-10 if values.size else 1e-12))[0]
        extra_rank = min(extra_rank, len(positive))
        if extra_rank > 0:
            singular = np.sqrt(values[:extra_rank])
            extras = (residual.T @ left[:, :extra_rank]) / (singular[None, :] + 1e-12)
            candidate = np.column_stack([q1, extras])
        else:
            candidate = q1[:, None]
    else:
        values = np.zeros((0,), dtype=np.float64)
        candidate = q1[:, None]

    q, _ = np.linalg.qr(candidate.astype(np.float64, copy=False))
    rank = min(int(max_rank), q.shape[1])
    basis = q[:, :rank].astype(np.float32)

    signed_mean_projection = xs @ basis[:, 0]
    metadata = {
        "method": "signed_matched_label_composition",
        "stored_rank": int(rank),
        "dimension": int(x.shape[1]),
        "n_contrasts": int(x.shape[0]),
        "signed_mean_norm": float(mean_norm),
        "signed_mean_projection_abs_mean": float(np.mean(np.abs(signed_mean_projection))),
        "residual_eigenvalues": values[: max(0, rank - 1)].tolist() if extra_rank > 0 else [],
    }
    return basis, metadata


def _filter_matrix(x: np.ndarray, basis: np.ndarray, rank: int, alpha: float) -> np.ndarray:
    rank = min(int(rank), int(basis.shape[1]))
    q = np.asarray(basis[:, :rank], dtype=np.float32)
    return (x - float(alpha) * (x @ q) @ q.T).astype(np.float32)


def _attack_metrics(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[float, float]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    tr = scaler.fit_transform(x_train)
    va = scaler.transform(x_val)
    clf = LogisticRegression(
        max_iter=4000,
        class_weight="balanced",
        random_state=20261007,
    )
    clf.fit(tr, y_train)
    pred = clf.predict(va)
    prob = clf.predict_proba(va)[:, 1]
    auc = float(roc_auc_score(y_val, prob))
    # A shadow-validation attacker can calibrate score orientation before target use.
    oriented_auc = max(auc, 1.0 - auc)
    ba = float(balanced_accuracy_score(y_val, pred))
    return oriented_auc, ba


def _select_head_operating_point(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    basis: np.ndarray,
) -> tuple[dict, list[dict]]:
    rows: list[dict] = []

    base_auc, base_ba = _attack_metrics(train_x, train_y, val_x, val_y)
    for rank in CANDIDATE_RANKS:
        if rank > basis.shape[1]:
            continue
        for alpha in CANDIDATE_ALPHAS:
            tx = _filter_matrix(train_x, basis, rank, alpha)
            vx = _filter_matrix(val_x, basis, rank, alpha)
            auc, ba = _attack_metrics(tx, train_y, vx, val_y)
            rel = np.linalg.norm(vx - val_x, axis=1) / (np.linalg.norm(val_x, axis=1) + 1e-12)
            rows.append(
                {
                    "rank": int(rank),
                    "alpha": float(alpha),
                    "shadow_val_oriented_auroc": float(auc),
                    "shadow_val_balanced_acc": float(ba),
                    "shadow_val_mean_relative_head_change": float(np.mean(rel)),
                    "undefended_shadow_val_oriented_auroc": float(base_auc),
                    "undefended_shadow_val_balanced_acc": float(base_ba),
                }
            )

    if not rows:
        raise RuntimeError("No valid classifier-head candidates were produced.")

    # Primary objective: minimize validation leakage. Distortion only breaks near-exact ties.
    rows.sort(
        key=lambda r: (
            round(r["shadow_val_oriented_auroc"], 6),
            r["shadow_val_mean_relative_head_change"],
            r["rank"],
            r["alpha"],
        )
    )
    return dict(rows[0]), rows


def _resolve_reference_run_dir(cfg: dict, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    output_root = Path(cfg["experiment"].get("output_root", "results/acm_revision/defense70_prep/reference"))
    return output_root / _run_id(cfg, "shadow_train")


def prepare(
    reference_config: str | Path,
    reference_run_dir: str | Path | None = None,
    output_dir: str | Path = "results/acm_revision/headfix_prep",
    samples_per_branch: int = 100,
) -> dict:
    cfg = copy.deepcopy(load_yaml(reference_config))
    cfg["seed"] = 142
    cfg["experiment"]["population"] = "shadow_train"
    cfg["experiment"]["setting_name"] = "modality_exclusive"
    cfg["experiment"]["concentration"] = 0.7
    cfg["defense"] = {"name": "none", "group": "classifier_head"}

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

    set_global_seed(142)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_run_dir = _resolve_reference_run_dir(cfg, str(reference_run_dir) if reference_run_dir else None)

    checkpoints = {
        r: reference_run_dir / "checkpoints" / f"round_{r:04d}.pt"
        for r in REFERENCE_ROUNDS
    }
    missing = [str(p) for p in checkpoints.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Head-fix preparation reuses the existing Defense70 reference checkpoints. "
            f"Missing checkpoint: {missing[0]}"
        )

    pools_dir = Path(cfg["data"]["pools_dir"])
    shadow_train = load_json(pools_dir / "shadow_train.json")
    shadow_val = load_json(pools_dir / "shadow_val.json")
    num_classes = int(cfg["data"]["num_classes"])
    specs = make_client_specs(12, "modality_exclusive", num_classes)

    model, tokenizer, image_processor = build_runtime(cfg)
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))

    contrasts: list[np.ndarray] = []
    contrast_labels: list[int] = []
    concentrated_rows: list[np.ndarray] = []
    concentrated_labels: list[int] = []
    concentrated_rounds: list[int] = []

    print("[HEADFIX 1/3] collecting signed matched classifier-head contrasts", flush=True)
    for round_id in REFERENCE_ROUNDS:
        state = torch.load(checkpoints[round_id], map_location="cpu")
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
                model,
                conc_samples,
                tokenizer,
                image_processor,
                spec.modality,
                cfg,
                device,
                pair_seed,
            )
            conc_vec = _exact_head_vector(before, conc_after)
            del conc_after

            bal_after, _ = local_train(
                model,
                bal_samples,
                tokenizer,
                image_processor,
                spec.modality,
                cfg,
                device,
                pair_seed,
            )
            bal_vec = _exact_head_vector(before, bal_after)
            del bal_after

            contrasts.append((conc_vec - bal_vec).astype(np.float32))
            contrast_labels.append(int(spec.dominant_label))
            concentrated_rows.append(conc_vec.astype(np.float32))
            concentrated_labels.append(int(spec.dominant_label))
            concentrated_rounds.append(int(round_id))

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del before, state
        gc.collect()

    basis, basis_meta = _fit_signed_basis(contrasts, contrast_labels, MAX_RANK)
    basis_path = output_dir / "signed_classifier_head_basis.npz"
    np.savez_compressed(basis_path, classifier_head=basis)

    print("[HEADFIX 2/3] collecting shadow-validation classifier-head updates", flush=True)
    val_clients, _ = partition_clients_strict(
        pool=shadow_val,
        specs=specs,
        num_classes=num_classes,
        concentration=0.7,
        samples_per_client=100,
        seed=242,
        partition_mode="fixed",
    )
    val_rows: list[np.ndarray] = []
    val_labels: list[int] = []
    val_rounds: list[int] = []

    for round_id in REFERENCE_ROUNDS:
        state = torch.load(checkpoints[round_id], map_location="cpu")
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
            val_rows.append(_exact_head_vector(before, after))
            val_labels.append(int(spec.dominant_label))
            val_rounds.append(int(round_id))
            del after
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del before, state
        gc.collect()

    tr_x_all = np.stack(concentrated_rows).astype(np.float32)
    tr_y_all = np.asarray(concentrated_labels, dtype=np.int64)
    tr_rounds = np.asarray(concentrated_rounds, dtype=np.int64)
    va_x_all = np.stack(val_rows).astype(np.float32)
    va_y_all = np.asarray(val_labels, dtype=np.int64)
    va_rounds = np.asarray(val_rounds, dtype=np.int64)

    train_mask = tr_rounds <= EARLY_MAX_ROUND
    val_mask = va_rounds > EARLY_MAX_ROUND
    selected, candidates = _select_head_operating_point(
        tr_x_all[train_mask],
        tr_y_all[train_mask],
        va_x_all[val_mask],
        va_y_all[val_mask],
        basis,
    )

    diagnostics_path = output_dir / "headfix_candidates.json"
    diagnostics_path.write_text(
        json.dumps(
            {
                "basis": basis_meta,
                "selection": candidates,
                "train_rounds": sorted(set(tr_rounds[train_mask].tolist())),
                "validation_rounds": sorted(set(va_rounds[val_mask].tolist())),
                "target_data_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = {
        "protocol": "HeadFix-Minimal-v1",
        "status": "LOCKED",
        "purpose": "minimal classifier-head repair for Defense70",
        "method": "signed_matched_label_composition_classifier_head",
        "scientific_scope": {
            "dataset": "hateful_memes",
            "setting": "modality_exclusive",
            "concentration": 0.7,
            "rounds": 150,
            "num_clients": 12,
        },
        "provenance": {
            "basis_population": "shadow_train",
            "basis_seed": 142,
            "validation_population": "shadow_val",
            "validation_seed": 242,
            "reference_rounds": list(REFERENCE_ROUNDS),
            "matched_contrast_samples_per_branch": int(samples_per_branch),
            "target_data_used_for_basis_or_tuning": False,
            "reference_run_dir": str(reference_run_dir),
        },
        "basis": {
            "path": str(basis_path),
            "group": "classifier_head",
            "stored_rank": int(basis.shape[1]),
            "dimension": int(basis.shape[0]),
            "signed_mean_first_direction": True,
            "residual_svd": True,
        },
        "selected": {
            "rank": int(selected["rank"]),
            "alpha": float(selected["alpha"]),
            "shadow_val_oriented_auroc": float(selected["shadow_val_oriented_auroc"]),
            "shadow_val_balanced_acc": float(selected["shadow_val_balanced_acc"]),
            "shadow_val_mean_relative_head_change": float(
                selected["shadow_val_mean_relative_head_change"]
            ),
        },
        "formal_run_populations": {
            "shadow_train": [142, 143],
            "shadow_val": [242],
            "target": [42, 43, 44, 45, 46],
        },
    }
    manifest_path = output_dir / "locked_headfix.yaml"
    with manifest_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True)

    print(
        "[HEADFIX 3/3] locked head fix: "
        f"rank={selected['rank']} alpha={selected['alpha']} "
        f"shadow-val oriented AUROC={selected['shadow_val_oriented_auroc']:.4f}",
        flush=True,
    )
    return {
        "status": "LOCKED",
        "manifest": str(manifest_path),
        "basis": str(basis_path),
        "selected": manifest["selected"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a minimal dominant-label-aware classifier-head repair."
    )
    parser.add_argument(
        "--reference-config",
        default="configs/acm_revision/generated/defense70_prep/reference_seed142.yaml",
    )
    parser.add_argument("--reference-run-dir", default=None)
    parser.add_argument(
        "--output-dir",
        default="results/acm_revision/headfix_prep",
    )
    parser.add_argument("--samples-per-branch", type=int, default=100)
    args = parser.parse_args()

    result = prepare(
        args.reference_config,
        args.reference_run_dir,
        args.output_dir,
        args.samples_per_branch,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
