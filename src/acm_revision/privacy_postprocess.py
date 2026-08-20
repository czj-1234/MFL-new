from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from .attacks import (
    load_matrix,
    make_attack_split,
    read_metadata,
    run_attack_grid,
    run_neural_trajectory_attack,
    run_trajectory_statistical_attack,
)


ALL_GROUPS = [
    "classifier_bias",
    "classifier_weight",
    "classifier_head",
    "fusion",
    "image_encoder",
    "text_encoder",
    "missing_modality",
    "all_shared",
    "full_update",
]


def _float_token(value: float | str) -> str:
    return str(value).replace(".", "p")


def _metadata_paths(root: Path, concentration: float, rounds: int, seeds: Sequence[int]) -> list[Path]:
    paths = []
    wanted_seeds = {int(x) for x in seeds}
    for path in root.rglob("update_metadata.csv"):
        summary_path = path.parent / "summary.json"
        capture_path = path.parent / "capture_manifest.json"
        if not summary_path.exists() or not capture_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            manifest = json.loads(capture_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if manifest.get("status") != "PASS":
            continue
        if int(summary.get("rounds", -1)) != int(rounds):
            continue
        if int(summary.get("seed", -1)) not in wanted_seeds:
            continue
        if not math.isclose(float(summary.get("concentration")), float(concentration), abs_tol=1e-9):
            continue
        paths.append(path)
    return sorted(paths)


def _validate_populations_and_seeds(frame: pd.DataFrame, seeds: Sequence[int]) -> dict:
    populations = set(frame["population"].astype(str).unique())
    required_populations = {"shadow_train", "shadow_val", "target"}
    missing_populations = sorted(required_populations - populations)
    observed_seeds = set(int(x) for x in frame["seed"].unique())
    missing_seeds = sorted(set(int(x) for x in seeds) - observed_seeds)
    if missing_populations or missing_seeds:
        raise ValueError(
            f"Incomplete privacy populations: missing_populations={missing_populations}, missing_seeds={missing_seeds}"
        )
    return {
        "populations": sorted(populations),
        "seeds": sorted(observed_seeds),
        "n_updates": int(len(frame)),
    }


def _available(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    column = f"has_{group}"
    if column in frame.columns:
        values = frame[column]
        if values.dtype == object:
            mask = values.astype(str).str.lower().isin({"true", "1", "yes"})
        else:
            mask = values.fillna(False).astype(bool)
        return frame[mask].copy()
    dim = f"dim_{group}"
    if dim in frame.columns:
        return frame[pd.to_numeric(frame[dim], errors="coerce").fillna(0) > 0].copy()
    return frame.copy()


def _compress(train: np.ndarray, val: np.ndarray, test: np.ndarray, pca_dim: int = 256):
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    val = scaler.transform(val)
    test = scaler.transform(test)
    max_dim = min(int(pca_dim), train.shape[0] - 1, train.shape[1])
    if max_dim > 0 and train.shape[1] > max_dim:
        pca = PCA(n_components=max_dim, random_state=42)
        train = pca.fit_transform(train)
        val = pca.transform(val)
        test = pca.transform(test)
    return train, val, test


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, prob: np.ndarray) -> dict:
    out = {
        "attack_acc": float(accuracy_score(y_true, y_pred)),
        "attack_asr": float(accuracy_score(y_true, y_pred)),
        "attack_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "attack_balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }
    try:
        out["attack_auroc"] = float(roc_auc_score(y_true, prob))
    except Exception:
        out["attack_auroc"] = float("nan")
    return out


def canonical_leakage_probabilities(
    frame: pd.DataFrame,
    group: str,
    output_dir: Path,
    pca_dim: int = 256,
) -> tuple[dict, pd.DataFrame]:
    subset = _available(frame, group)
    split = make_attack_split(subset, "cross_partition", seed=42)
    train, val, test = split.train.copy(), split.val.copy(), split.test.copy()
    x_train = load_matrix(train, group, "observed")
    x_val = load_matrix(val, group, "observed")
    x_test = load_matrix(test, group, "observed")
    x_train, x_val, x_test = _compress(x_train, x_val, x_test, pca_dim=pca_dim)
    y_train = train["dominant_label"].to_numpy(dtype=int)
    y_val = val["dominant_label"].to_numpy(dtype=int)
    y_test = test["dominant_label"].to_numpy(dtype=int)

    candidates = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_c = candidates[0]
    best_score = -float("inf")
    for c in candidates:
        model = LogisticRegression(max_iter=4000, class_weight="balanced", C=c, random_state=42)
        model.fit(x_train, y_train)
        val_prob = model.predict_proba(x_val)[:, 1]
        try:
            score = float(roc_auc_score(y_val, val_prob))
        except Exception:
            score = float(accuracy_score(y_val, model.predict(x_val)))
        if score > best_score:
            best_score = score
            best_c = c

    model = LogisticRegression(max_iter=4000, class_weight="balanced", C=best_c, random_state=42)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)
    predictions = model.predict(x_test)
    metrics = _binary_metrics(y_test, predictions, probabilities[:, 1])
    metrics.update(
        {
            "group": group,
            "protocol": "cross_partition",
            "attack_model": "logistic_regression_tuned_on_shadow_val",
            "selected_C": float(best_c),
            "shadow_val_selection_score": float(best_score),
            "n_train": int(len(train)),
            "n_val": int(len(val)),
            "n_test": int(len(test)),
            "feature_dim": int(x_train.shape[1]),
        }
    )

    predictions_frame = test[
        [
            "run_id",
            "population",
            "seed",
            "round",
            "client_id",
            "client_key",
            "concentration",
            "modality",
            "dominant_label",
            "update_path",
        ]
    ].copy()
    predictions_frame["group"] = group
    predictions_frame["predicted_dominant_label"] = predictions
    predictions_frame["prob_label_0"] = probabilities[:, 0]
    predictions_frame["prob_label_1"] = probabilities[:, 1]
    predictions_frame["leakage_probability_true_label"] = np.where(
        y_test == 1, probabilities[:, 1], probabilities[:, 0]
    )
    predictions_frame["attack_confidence"] = probabilities.max(axis=1)
    predictions_frame.to_csv(output_dir / f"leakage_probabilities__{group}.csv", index=False)
    return metrics, predictions_frame


def label_distribution_inference(frame: pd.DataFrame, group: str, output_dir: Path, pca_dim: int = 256) -> dict:
    subset = _available(frame, group)
    split = make_attack_split(subset, "cross_partition", seed=42)
    train, val, test = split.train.copy(), split.val.copy(), split.test.copy()
    x_train = load_matrix(train, group, "observed")
    x_val = load_matrix(val, group, "observed")
    x_test = load_matrix(test, group, "observed")
    x_train, x_val, x_test = _compress(x_train, x_val, x_test, pca_dim=pca_dim)
    label_columns = ["label_prop_0", "label_prop_1"]
    y_train = train[label_columns].to_numpy(dtype=np.float32)
    y_val = val[label_columns].to_numpy(dtype=np.float32)
    y_test = test[label_columns].to_numpy(dtype=np.float32)

    alpha_candidates = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_alpha = alpha_candidates[0]
    best_mae = float("inf")
    for alpha in alpha_candidates:
        pred_val = []
        for cls in range(2):
            model = Ridge(alpha=alpha).fit(x_train, y_train[:, cls])
            pred_val.append(model.predict(x_val))
        pred_val = np.stack(pred_val, axis=1)
        pred_val = np.maximum(pred_val, 0)
        pred_val /= pred_val.sum(axis=1, keepdims=True) + 1e-12
        mae = float(mean_absolute_error(y_val, pred_val))
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha

    pred_test = []
    for cls in range(2):
        model = Ridge(alpha=best_alpha).fit(x_train, y_train[:, cls])
        pred_test.append(model.predict(x_test))
    pred_test = np.stack(pred_test, axis=1)
    pred_test = np.maximum(pred_test, 0)
    pred_test /= pred_test.sum(axis=1, keepdims=True) + 1e-12
    true_dom = y_test.argmax(axis=1)
    pred_dom = pred_test.argmax(axis=1)
    metrics = _binary_metrics(true_dom, pred_dom, pred_test[:, 1])
    metrics.update(
        {
            "group": group,
            "attack_model": "ridge_label_distribution_then_argmax",
            "selected_alpha": float(best_alpha),
            "shadow_val_distribution_mae": float(best_mae),
            "target_distribution_mae": float(mean_absolute_error(y_test, pred_test)),
            "n_train": int(len(train)),
            "n_val": int(len(val)),
            "n_test": int(len(test)),
        }
    )

    output = test[
        ["run_id", "seed", "round", "client_id", "client_key", "concentration", "dominant_label"]
    ].copy()
    output["group"] = group
    output["true_prop_0"] = y_test[:, 0]
    output["true_prop_1"] = y_test[:, 1]
    output["pred_prop_0"] = pred_test[:, 0]
    output["pred_prop_1"] = pred_test[:, 1]
    output["predicted_dominant_label"] = pred_dom
    output.to_csv(output_dir / f"label_distribution_predictions__{group}.csv", index=False)
    return metrics


def _pairwise_cosines(x: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    normed = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    sim = normed @ normed.T
    within = []
    between = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            (within if labels[i] == labels[j] else between).append(float(sim[i, j]))
    within_mean = float(np.mean(within)) if within else float("nan")
    between_mean = float(np.mean(between)) if between else float("nan")
    gap = within_mean - between_mean if within and between else float("nan")
    return within_mean, between_mean, gap


def _svd_stats(x: np.ndarray) -> dict:
    singular = np.linalg.svd(x, compute_uv=False)
    energy = singular**2
    total = float(energy.sum() + 1e-12)
    p = singular / (singular.sum() + 1e-12)
    effective_rank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
    return {
        "effective_rank": effective_rank,
        "top1_energy": float(energy[:1].sum() / total),
        "top3_energy": float(energy[: min(3, len(energy))].sum() / total),
        "top5_energy": float(energy[: min(5, len(energy))].sum() / total),
        "top10_energy": float(energy[: min(10, len(energy))].sum() / total),
    }


def structural_privacy(frame: pd.DataFrame, groups: Sequence[str], output_path: Path) -> pd.DataFrame:
    rows = []
    for group in groups:
        subset = _available(frame, group)
        if subset.empty:
            continue
        for keys, part in subset.groupby(["population", "concentration", "round"], dropna=False):
            try:
                x = load_matrix(part, group, "observed")
                labels = part["dominant_label"].to_numpy(dtype=int)
                norms = np.linalg.norm(x, axis=1)
                within, between, gap = _pairwise_cosines(x, labels)
                row = {
                    "population": keys[0],
                    "concentration": keys[1],
                    "round": int(keys[2]),
                    "group": group,
                    "representation": (
                        str(part[f"repr_{group}"].dropna().iloc[0])
                        if f"repr_{group}" in part and part[f"repr_{group}"].notna().any()
                        else "legacy_or_exact"
                    ),
                    "n_updates": int(len(part)),
                    "n_independent_seeds": int(part["seed"].nunique()),
                    "update_norm_mean": float(norms.mean()),
                    "update_norm_std": float(norms.std(ddof=1)) if len(norms) > 1 else 0.0,
                    "within_label_cosine_mean": within,
                    "between_label_cosine_mean": between,
                    "cosine_separation_gap": gap,
                    **_svd_stats(x),
                }
                unique, counts = np.unique(labels, return_counts=True)
                valid_clusters = len(unique) >= 2 and len(x) > len(unique) and counts.min() >= 2
                if valid_clusters:
                    row["silhouette_score"] = float(silhouette_score(x, labels, metric="euclidean"))
                    row["davies_bouldin_index"] = float(davies_bouldin_score(x, labels))
                    row["calinski_harabasz_index"] = float(calinski_harabasz_score(x, labels))
                else:
                    row["silhouette_score"] = float("nan")
                    row["davies_bouldin_index"] = float("nan")
                    row["calinski_harabasz_index"] = float("nan")
                rows.append(row)
            except Exception as exc:
                rows.append(
                    {
                        "population": keys[0],
                        "concentration": keys[1],
                        "round": int(keys[2]),
                        "group": group,
                        "error": repr(exc),
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    return result


def run_trajectory_suite(frame: pd.DataFrame, output_dir: Path, smoke: bool = False) -> pd.DataFrame:
    subset = _available(frame, "classifier_head")
    split = make_attack_split(subset, "cross_partition", seed=42)
    rows = []
    statistical = ["temporal_mean", "temporal_variance", "mean_variance", "concatenation"]
    for method in statistical:
        try:
            rows.append(
                run_trajectory_statistical_attack(
                    split,
                    "classifier_head",
                    method,
                    target="dominant_label",
                    observation="observed",
                    sequence_length=min(10, int(subset["round"].nunique())),
                    seed=42,
                )
            )
        except Exception as exc:
            rows.append({"trajectory_method": method, "error": repr(exc)})
    if not smoke:
        for architecture in ["lstm", "tcn", "transformer"]:
            try:
                rows.append(
                    run_neural_trajectory_attack(
                        split,
                        "classifier_head",
                        architecture,
                        target="dominant_label",
                        observation="observed",
                        sequence_length=10,
                        epochs=100,
                        seed=42,
                    )
                )
            except Exception as exc:
                rows.append({"trajectory_method": architecture, "error": repr(exc)})
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "trajectory_attacks.csv", index=False)
    return result


def run_postprocess(
    root: str | Path,
    concentration: float,
    rounds: int,
    seeds: Sequence[int],
    output_root: str | Path | None = None,
    smoke: bool = False,
) -> dict:
    root = Path(root)
    metadata_paths = _metadata_paths(root, concentration, rounds, seeds)
    if not metadata_paths:
        raise FileNotFoundError(
            f"No validated privacy metadata found under {root} for c={concentration}, rounds={rounds}, seeds={list(seeds)}"
        )
    frame = read_metadata(metadata_paths)
    validation = _validate_populations_and_seeds(frame, seeds)

    output_dir = Path(output_root or (root / "postprocess")) / f"c{_float_token(concentration)}_r{rounds}"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "combined_update_metadata.csv", index=False)

    groups = ALL_GROUPS if not smoke else ["classifier_head", "fusion", "full_update"]
    attack_models = (
        ["logistic_regression"]
        if smoke
        else [
            "random",
            "majority",
            "update_norm",
            "cosine_centroid",
            "logistic_regression",
            "linear_svm",
            "random_forest",
            "boosted_trees",
            "xgboost",
            "mlp",
        ]
    )
    protocols = ["cross_partition"] if smoke else [
        "random_update",
        "temporal",
        "cross_client",
        "cross_partition",
        "cross_partition_temporal",
    ]

    attack_outputs = []
    canonical_rows = []
    label_distribution_rows = []
    for group in groups:
        group_frame = _available(frame, group)
        if group_frame.empty:
            continue
        group_metadata = output_dir / f"metadata__{group}.csv"
        group_frame.to_csv(group_metadata, index=False)
        attack_path = output_dir / f"attack_summary__{group}.csv"
        run_attack_grid(
            [group_metadata],
            attack_path,
            protocols=protocols,
            groups=[group],
            models=attack_models,
            targets=["dominant_label", "modality"] if not smoke else ["dominant_label"],
            observation="observed",
            seed=42,
            pca_dim=256,
        )
        attack_outputs.append(str(attack_path))
        try:
            metrics, _ = canonical_leakage_probabilities(frame, group, output_dir)
            canonical_rows.append(metrics)
        except Exception as exc:
            canonical_rows.append({"group": group, "error": repr(exc)})
        try:
            label_distribution_rows.append(label_distribution_inference(frame, group, output_dir))
        except Exception as exc:
            label_distribution_rows.append({"group": group, "error": repr(exc)})

    canonical_frame = pd.DataFrame(canonical_rows)
    canonical_frame.to_csv(output_dir / "canonical_leakage_summary.csv", index=False)
    label_frame = pd.DataFrame(label_distribution_rows)
    label_frame.to_csv(output_dir / "label_distribution_summary.csv", index=False)
    structural = structural_privacy(frame, groups, output_dir / "structural_privacy.csv")
    trajectory = run_trajectory_suite(frame, output_dir, smoke=smoke)

    required = [
        output_dir / "combined_update_metadata.csv",
        output_dir / "canonical_leakage_summary.csv",
        output_dir / "label_distribution_summary.csv",
        output_dir / "structural_privacy.csv",
        output_dir / "trajectory_attacks.csv",
    ]
    failures = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
    report = {
        "status": "PASS" if not failures else "FAIL",
        "root": str(root),
        "output_dir": str(output_dir),
        "concentration": float(concentration),
        "rounds": int(rounds),
        "seeds": [int(x) for x in seeds],
        "validation": validation,
        "metadata_paths": [str(x) for x in metadata_paths],
        "attack_outputs": attack_outputs,
        "n_structural_rows": int(len(structural)),
        "n_trajectory_rows": int(len(trajectory)),
        "failures": failures,
        "representation_note": (
            "classifier and fusion groups are captured exactly; large encoder/all-shared/full-update groups "
            "use deterministic coordinate sketches at milestone rounds."
        ),
    }
    (output_dir / "postprocess_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if failures:
        raise RuntimeError(f"Postprocess validation failed: {failures}")
    return report


def _parse_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete leakage and structural-privacy post-processing")
    parser.add_argument("--root", required=True)
    parser.add_argument("--concentration", type=float, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    report = run_postprocess(
        args.root,
        args.concentration,
        args.rounds,
        _parse_ints(args.seeds),
        output_root=args.output_root,
        smoke=args.smoke,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
