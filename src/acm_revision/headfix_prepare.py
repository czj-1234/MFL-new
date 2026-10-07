from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .attacks import load_matrix, read_metadata


TRAIN_SEEDS = (142, 143)
VAL_SEEDS = (242,)
SETTING = "modality_exclusive"
CONCENTRATION = 0.7
ROUNDS = 150
TRAIN_MAX_ROUND = 60
VAL_MIN_ROUND = 61
GROUP = "classifier_head"
MAX_RANK = 64
EVAL_RANKS = (0, 1, 2, 3, 5, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64)


def _discover_metadata(
    core_root: str | Path,
    setting: str = SETTING,
    concentration: float = CONCENTRATION,
    rounds: int = ROUNDS,
) -> list[Path]:
    root = Path(core_root)
    wanted = {
        "shadow_train": set(TRAIN_SEEDS),
        "shadow_val": set(VAL_SEEDS),
    }
    paths: list[Path] = []
    for path in root.rglob("update_metadata.csv"):
        summary_path = path.parent / "summary.json"
        capture_path = path.parent / "capture_manifest.json"
        if not summary_path.exists() or not capture_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            capture = json.loads(capture_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        pop = str(summary.get("population"))
        if pop not in wanted:
            continue
        if capture.get("status") != "PASS":
            continue
        if str(summary.get("setting_name")) != setting:
            continue
        if int(summary.get("rounds", -1)) != int(rounds):
            continue
        if int(summary.get("seed", -1)) not in wanted[pop]:
            continue
        if not math.isclose(
            float(summary.get("concentration")),
            float(concentration),
            abs_tol=1e-9,
        ):
            continue
        paths.append(path)
    return sorted(paths)


def _validate_frame(frame: pd.DataFrame) -> dict:
    required = {
        "population",
        "seed",
        "round",
        "dominant_label",
        "update_path",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")

    failures: list[str] = []
    got_train = sorted(
        int(x)
        for x in frame.loc[
            frame["population"].astype(str) == "shadow_train", "seed"
        ].unique()
    )
    got_val = sorted(
        int(x)
        for x in frame.loc[
            frame["population"].astype(str) == "shadow_val", "seed"
        ].unique()
    )
    if got_train != list(TRAIN_SEEDS):
        failures.append(f"shadow_train seeds expected {TRAIN_SEEDS}, got {got_train}")
    if got_val != list(VAL_SEEDS):
        failures.append(f"shadow_val seeds expected {VAL_SEEDS}, got {got_val}")

    train = frame[
        (frame["population"].astype(str) == "shadow_train")
        & (frame["round"].astype(int) <= TRAIN_MAX_ROUND)
    ].copy()
    val = frame[
        (frame["population"].astype(str) == "shadow_val")
        & (frame["round"].astype(int) >= VAL_MIN_ROUND)
    ].copy()

    if train.empty or val.empty:
        failures.append(
            f"strict temporal split empty: train={len(train)} val={len(val)}"
        )
    if train["dominant_label"].nunique() != 2:
        failures.append("shadow_train strict split does not contain two dominant labels")
    if val["dominant_label"].nunique() != 2:
        failures.append("shadow_val strict split does not contain two dominant labels")

    if failures:
        raise RuntimeError("; ".join(failures))

    return {
        "status": "PASS",
        "n_train_updates": int(len(train)),
        "n_val_updates": int(len(val)),
        "train_seeds": got_train,
        "val_seeds": got_val,
        "train_round_max": int(train["round"].max()),
        "val_round_min": int(val["round"].min()),
    }


def _project(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    if basis.size == 0:
        return np.asarray(x, dtype=np.float32).copy()
    q = np.asarray(basis, dtype=np.float32)
    return (x - (x @ q) @ q.T).astype(np.float32)


def _orthogonalize(direction: np.ndarray, basis: np.ndarray) -> np.ndarray:
    d = np.asarray(direction, dtype=np.float64).reshape(-1)
    if basis.size:
        q = np.asarray(basis, dtype=np.float64)
        d = d - q @ (q.T @ d)
    norm = float(np.linalg.norm(d))
    if norm <= 1e-12:
        return np.zeros_like(d, dtype=np.float32)
    return (d / norm).astype(np.float32)


def _fit_lr_direction(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Fit a linear dominant-label attacker and map its direction back to raw coordinates."""
    scaler = StandardScaler(with_mean=True, with_std=True)
    xs = scaler.fit_transform(x)
    clf = LogisticRegression(
        max_iter=4000,
        class_weight="balanced",
        random_state=int(seed),
    )
    clf.fit(xs, y)
    coef = np.asarray(clf.coef_[0], dtype=np.float64)
    raw_direction = coef / (np.asarray(scaler.scale_, dtype=np.float64) + 1e-12)
    return raw_direction.astype(np.float32)


def _prepare_attack_features(
    x_train: np.ndarray,
    x_val: np.ndarray,
    pca_dim: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    tr = scaler.fit_transform(x_train)
    va = scaler.transform(x_val)
    max_dim = min(int(pca_dim), tr.shape[0] - 1, tr.shape[1])
    if max_dim > 0 and tr.shape[1] > max_dim:
        pca = PCA(n_components=max_dim, random_state=42)
        tr = pca.fit_transform(tr)
        va = pca.transform(va)
    return tr, va


def _oriented_auc(y: np.ndarray, prob: np.ndarray) -> float:
    auc = float(roc_auc_score(y, prob))
    return max(auc, 1.0 - auc)


def _evaluate_attack_family(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 20261007,
) -> dict:
    tr, va = _prepare_attack_features(x_train, x_val, pca_dim=256)
    rows: dict[str, dict] = {}

    models = {
        "LR": LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            random_state=seed,
        ),
        "RF": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=500,
            early_stopping=True,
            random_state=seed,
        ),
    }

    for name, model in models.items():
        model.fit(tr, y_train)
        pred = model.predict(va)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(va)[:, 1]
        else:
            score = np.asarray(model.decision_function(va), dtype=np.float64)
            prob = score
        rows[name] = {
            "oriented_auroc": float(_oriented_auc(y_val, prob)),
            "balanced_acc": float(balanced_accuracy_score(y_val, pred)),
        }

    strongest = max(rows, key=lambda k: rows[k]["oriented_auroc"])
    return {
        "attacks": rows,
        "strongest_attack": strongest,
        "worst_oriented_auroc": float(rows[strongest]["oriented_auroc"]),
    }


def _relative_change(original: np.ndarray, filtered: np.ndarray) -> float:
    num = np.linalg.norm(filtered - original, axis=1)
    den = np.linalg.norm(original, axis=1) + 1e-12
    return float(np.mean(num / den))


def _build_iterative_basis(
    x_train: np.ndarray,
    y_train: np.ndarray,
    max_rank: int,
) -> tuple[np.ndarray, list[dict]]:
    basis = np.zeros((x_train.shape[1], 0), dtype=np.float32)
    trace: list[dict] = []

    for rank in range(1, int(max_rank) + 1):
        projected = _project(x_train, basis)
        raw_direction = _fit_lr_direction(projected, y_train, seed=20261007 + rank)
        direction = _orthogonalize(raw_direction, basis)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-12:
            print(
                f"[HEADFIX BASIS] stopped at rank={rank-1}: no independent LR direction remains",
                flush=True,
            )
            break
        basis = np.column_stack([basis, direction]).astype(np.float32)
        trace.append(
            {
                "rank": int(rank),
                "direction_norm_after_orthogonalization": direction_norm,
            }
        )
        print(
            f"[HEADFIX BASIS] learned direction {rank}/{max_rank}",
            flush=True,
        )

    if basis.shape[1] < 1:
        raise RuntimeError("No attacker-guided classifier-head direction was learned.")
    return basis, trace


def prepare(
    core_root: str | Path = "results/acm_revision/core72_r150",
    output_dir: str | Path = "results/acm_revision/headfix_prep",
    max_rank: int = MAX_RANK,
    target_auc: float = 0.70,
    launch_auc: float = 0.80,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_paths = _discover_metadata(core_root)
    if not metadata_paths:
        raise FileNotFoundError(
            f"No matching Core72 metadata found under {core_root} for "
            f"{SETTING}, c={CONCENTRATION}, rounds={ROUNDS}."
        )

    frame = read_metadata(metadata_paths)
    validation = _validate_frame(frame)

    train = frame[
        (frame["population"].astype(str) == "shadow_train")
        & (frame["round"].astype(int) <= TRAIN_MAX_ROUND)
    ].copy()
    val = frame[
        (frame["population"].astype(str) == "shadow_val")
        & (frame["round"].astype(int) >= VAL_MIN_ROUND)
    ].copy()

    print(
        "[HEADFIX 1/3] loading existing exact classifier-head updates "
        f"(train={len(train)}, val={len(val)})",
        flush=True,
    )
    x_train = load_matrix(train, GROUP, "observed").astype(np.float32)
    y_train = train["dominant_label"].to_numpy(dtype=np.int64)
    x_val = load_matrix(val, GROUP, "observed").astype(np.float32)
    y_val = val["dominant_label"].to_numpy(dtype=np.int64)

    if x_train.shape[1] != x_val.shape[1]:
        raise ValueError(
            f"Head dimension mismatch: train={x_train.shape[1]} val={x_val.shape[1]}"
        )

    print(
        f"[HEADFIX 2/3] iterative attacker-guided nullspace search "
        f"(dimension={x_train.shape[1]}, max_rank={max_rank})",
        flush=True,
    )
    basis, direction_trace = _build_iterative_basis(x_train, y_train, max_rank)

    basis_path = output_dir / "attacker_guided_classifier_head_basis.npz"
    np.savez_compressed(basis_path, classifier_head=basis)

    eval_rows: list[dict] = []
    ranks_to_eval = [r for r in EVAL_RANKS if r <= basis.shape[1]]
    if basis.shape[1] not in ranks_to_eval:
        ranks_to_eval.append(int(basis.shape[1]))
    ranks_to_eval = sorted(set(ranks_to_eval))

    print("[HEADFIX 3/3] evaluating LR/RF/MLP on shadow validation", flush=True)
    for rank in ranks_to_eval:
        if rank == 0:
            tr_f = x_train
            va_f = x_val
        else:
            q = basis[:, :rank]
            tr_f = _project(x_train, q)
            va_f = _project(x_val, q)

        metrics = _evaluate_attack_family(tr_f, y_train, va_f, y_val)
        distortion = _relative_change(x_val, va_f)
        row = {
            "rank": int(rank),
            "alpha": 1.0,
            "worst_oriented_auroc": float(metrics["worst_oriented_auroc"]),
            "strongest_attack": str(metrics["strongest_attack"]),
            "shadow_val_mean_relative_head_change": float(distortion),
            "attack_metrics": metrics["attacks"],
        }
        eval_rows.append(row)
        print(
            "[HEADFIX EVAL] "
            f"rank={rank:>2} "
            f"worstAUC={row['worst_oriented_auroc']:.4f} "
            f"strongest={row['strongest_attack']} "
            f"relChange={distortion:.4f}",
            flush=True,
        )

    defended_rows = [r for r in eval_rows if r["rank"] > 0]
    feasible = [r for r in defended_rows if r["worst_oriented_auroc"] <= float(target_auc)]
    if feasible:
        selected = min(
            feasible,
            key=lambda r: (
                r["rank"],
                r["shadow_val_mean_relative_head_change"],
                r["worst_oriented_auroc"],
            ),
        )
        selection_reason = f"smallest evaluated rank reaching worst AUC <= {target_auc:.3f}"
    else:
        selected = min(
            defended_rows,
            key=lambda r: (
                r["worst_oriented_auroc"],
                r["shadow_val_mean_relative_head_change"],
                r["rank"],
            ),
        )
        selection_reason = "minimum shadow-validation worst AUC among evaluated ranks"

    status = (
        "LOCKED"
        if float(selected["worst_oriented_auroc"]) <= float(launch_auc)
        else "REJECTED"
    )

    diagnostics = {
        "protocol": "HeadFix-Minimal-v2",
        "validation": validation,
        "metadata_paths": [str(p) for p in metadata_paths],
        "target_data_used": False,
        "basis_learning": {
            "method": "iterative_attacker_guided_nullspace",
            "group": GROUP,
            "dimension": int(x_train.shape[1]),
            "stored_rank": int(basis.shape[1]),
            "direction_trace": direction_trace,
        },
        "evaluation": {
            "attack_family": ["LR", "RF", "MLP"],
            "strict_train": "shadow_train seeds 142,143; rounds <= 60",
            "strict_validation": "shadow_val seed 242; rounds >= 61",
            "pca_dim": 256,
            "rows": eval_rows,
        },
        "selection_reason": selection_reason,
    }
    (output_dir / "headfix_candidates.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "protocol": "HeadFix-Minimal-v2",
        "status": status,
        "purpose": "minimal classifier-head repair for Defense70",
        "method": "iterative_attacker_guided_nullspace_classifier_head",
        "scientific_scope": {
            "dataset": "hateful_memes",
            "setting": SETTING,
            "concentration": CONCENTRATION,
            "rounds": ROUNDS,
            "num_clients": 12,
        },
        "provenance": {
            "basis_population": "shadow_train",
            "basis_seeds": list(TRAIN_SEEDS),
            "basis_rounds": f"1-{TRAIN_MAX_ROUND}",
            "validation_population": "shadow_val",
            "validation_seeds": list(VAL_SEEDS),
            "validation_rounds": f"{VAL_MIN_ROUND}-{ROUNDS}",
            "target_data_used_for_basis_or_tuning": False,
            "source_root": str(core_root),
        },
        "basis": {
            "path": str(basis_path),
            "group": GROUP,
            "stored_rank": int(basis.shape[1]),
            "dimension": int(basis.shape[0]),
            "construction": "iterative LR direction removal with refitting after each projection",
        },
        "selection": {
            "target_shadow_val_worst_auc": float(target_auc),
            "formal_launch_threshold": float(launch_auc),
            "reason": selection_reason,
        },
        "selected": {
            "rank": int(selected["rank"]),
            "alpha": 1.0,
            "shadow_val_worst_oriented_auroc": float(
                selected["worst_oriented_auroc"]
            ),
            "shadow_val_strongest_attack": str(selected["strongest_attack"]),
            "shadow_val_mean_relative_head_change": float(
                selected["shadow_val_mean_relative_head_change"]
            ),
            "attack_metrics": selected["attack_metrics"],
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

    if status == "LOCKED":
        print(
            "[HEADFIX READY] "
            f"rank={selected['rank']} "
            f"worst shadow-val AUC={selected['worst_oriented_auroc']:.4f} "
            f"strongest={selected['strongest_attack']} "
            f"relChange={selected['shadow_val_mean_relative_head_change']:.4f}",
            flush=True,
        )
    else:
        print(
            "[HEADFIX STOP] formal FL runs NOT recommended: "
            f"best worst shadow-val AUC={selected['worst_oriented_auroc']:.4f} "
            f"> launch threshold {launch_auc:.4f}",
            flush=True,
        )

    return {
        "status": status,
        "manifest": str(manifest_path),
        "basis": str(basis_path),
        "selected": manifest["selected"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an iterative attacker-guided classifier-head nullspace using "
            "existing Core72 shadow updates only; no CLIP retraining is performed."
        )
    )
    parser.add_argument(
        "--core-root",
        default="results/acm_revision/core72_r150",
    )
    parser.add_argument(
        "--output-dir",
        default="results/acm_revision/headfix_prep",
    )
    parser.add_argument("--max-rank", type=int, default=MAX_RANK)
    parser.add_argument("--target-auc", type=float, default=0.70)
    parser.add_argument("--launch-auc", type=float, default=0.80)
    args = parser.parse_args()

    result = prepare(
        core_root=args.core_root,
        output_dir=args.output_dir,
        max_rank=args.max_rank,
        target_auc=args.target_auc,
        launch_auc=args.launch_auc,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
