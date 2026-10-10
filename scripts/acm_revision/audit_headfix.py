#!/usr/bin/env python3
"""Shadow-validation-selected all-surface HeadFix leakage and utility audit.

Run from the MFL-new repository root. Never retrains a federated model.
"""
from __future__ import annotations

import argparse
import json
import math
import tarfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

GROUPS = (
    "classifier_bias", "classifier_weight", "classifier_head", "fusion",
    "missing_modality", "image_encoder", "text_encoder", "all_shared", "full_update",
)
PROTOCOLS = ("cross_partition", "cross_partition_temporal")
EXPECTED = {"shadow_train": (142, 143), "shadow_val": (242,), "target": (42, 43, 44, 45, 46)}
SEED = 42


def bootstrap(values, nboot=5000):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0, "mean": None, "std": None, "ci_low": None, "ci_high": None}
    if len(x) < 2:
        return {"n": len(x), "mean": float(x[0]), "std": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(20261010)
    sampled = rng.choice(x, size=(nboot, len(x)), replace=True).mean(axis=1)
    return {"n": int(len(x)), "mean": float(x.mean()), "std": float(x.std(ddof=1)),
            "ci_low": float(np.quantile(sampled, .025)),
            "ci_high": float(np.quantile(sampled, .975))}


def discover_runs(root):
    valid = []
    for metadata_path in sorted(root.rglob("update_metadata.csv")):
        summary_path = metadata_path.parent / "summary.json"
        capture_path = metadata_path.parent / "capture_manifest.json"
        if not summary_path.exists() or not capture_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        capture = json.loads(capture_path.read_text(encoding="utf-8"))
        pop = str(summary.get("population"))
        seed = int(summary.get("seed", -1))
        if (pop not in EXPECTED or seed not in EXPECTED[pop]
                or summary.get("setting_name") != "modality_exclusive"
                or not math.isclose(float(summary.get("concentration", -1)), .7, abs_tol=1e-9)):
            continue
        if int(summary.get("rounds", -1)) != 150 or capture.get("status") != "PASS":
            raise ValueError(f"Incomplete run or failed capture: {metadata_path.parent}")
        metrics = pd.read_csv(metadata_path.parent / "round_metrics.csv")
        if metrics.empty or int(metrics["round"].max()) != 150:
            raise ValueError(f"Run did not reach round 150: {metadata_path.parent}")
        valid.append((metadata_path, summary))
    got = [(s["population"], int(s["seed"])) for _, s in valid]
    required = {(pop, seed) for pop, seeds in EXPECTED.items() for seed in seeds}
    if len(got) != len(required) or set(got) != required:
        raise ValueError(f"Expected exactly 8 unique runs; got={got}, missing={required-set(got)}")
    return valid


def all_metadata(runs):
    frames = []
    for path, summary in runs:
        df = pd.read_csv(path)
        if df.empty or not ((df["population"].astype(str) == summary["population"]).all()
                            and (df["seed"].astype(int) == int(summary["seed"])).all()):
            raise ValueError(f"Incorrect metadata: {path}")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def group_metadata(frame, group):
    flag = f"has_{group}"
    if flag in frame.columns:
        col = frame[flag]
        if col.dtype == object:
            mask = col.astype(str).str.lower().isin(("true", "1", "yes"))
        else:
            mask = col.fillna(False).astype(bool)
    elif f"dim_{group}" in frame.columns:
        mask = pd.to_numeric(frame[f"dim_{group}"], errors="coerce").fillna(0) > 0
    else:
        mask = pd.Series(True, index=frame.index)
    return frame.loc[mask].copy().reset_index(drop=True)


def read_group_vectors(frame, group):
    arrays = []
    for path_value in frame["update_path"]:
        path = Path(str(path_value))
        if not path.exists():
            raise FileNotFoundError(
                f"Missing update file: {path}. Run from the repository root, with original results available."
            )
        with np.load(path, allow_pickle=False) as data:
            key = f"observed__{group}"
            if key not in data.files:
                raise KeyError(f"{key} missing in {path}")
            arrays.append(np.asarray(data[key], dtype=np.float32).reshape(-1))
    dims = {len(x) for x in arrays}
    if len(dims) != 1:
        raise ValueError(f"{group} inconsistent feature dimensions: {dims}")
    return np.stack(arrays)


def get_splits(frame, protocol):
    population = frame["population"].astype(str).to_numpy()
    rounds = frame["round"].to_numpy(dtype=int)
    if protocol == "cross_partition":
        return (np.where(population == "shadow_train")[0],
                np.where(population == "shadow_val")[0],
                np.where(population == "target")[0])
    if protocol == "cross_partition_temporal":
        return (np.where((population == "shadow_train") & (rounds <= 60))[0],
                np.where((population == "shadow_val") & (rounds > 60))[0],
                np.where((population == "target") & (rounds >= 90))[0])
    raise ValueError(protocol)


def features(train, validation, target):
    scaler = StandardScaler()
    a = scaler.fit_transform(train)
    b = scaler.transform(validation)
    c = scaler.transform(target)
    d = min(256, a.shape[0] - 1, a.shape[1])
    if d > 0 and a.shape[1] > d:
        pca = PCA(n_components=d, random_state=SEED)
        a = pca.fit_transform(a)
        b = pca.transform(b)
        c = pca.transform(c)
    return a, b, c


def attacker(name):
    if name == "LR":
        return LogisticRegression(max_iter=4000, class_weight="balanced", random_state=SEED)
    if name == "RF":
        return RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                      random_state=SEED, n_jobs=4)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500,
                             early_stopping=True, random_state=SEED)
    raise ValueError(name)


def auroc(y, score):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def evaluate(group, protocol, model_name, train_ids, val_ids, test_ids, labels, meta, x):
    xtr, xval, xtest = features(x[train_ids], x[val_ids], x[test_ids])
    model = attacker(model_name)
    model.fit(xtr, labels[train_ids])
    sv = model.predict_proba(xval)[:, 1]
    st = model.predict_proba(xtest)[:, 1]
    auc_val = auroc(labels[val_ids], sv)
    if not np.isfinite(auc_val):
        raise RuntimeError(f"No shadow validation AUROC: {group}/{protocol}/{model_name}")
    flip = auc_val < .5
    auc_target = auroc(labels[test_ids], st)
    row = {"group": group, "protocol": protocol, "model": model_name,
           "shadow_val_auc_raw": auc_val,
           "shadow_val_auc_oriented": max(auc_val, 1 - auc_val),
           "orientation_flip_selected_on_shadow_val": bool(flip),
           "target_auc_raw": auc_target,
           "target_auc_validation_oriented": (1 - auc_target if flip else auc_target),
           "n_train": len(train_ids), "n_shadow_val": len(val_ids),
           "n_target": len(test_ids), "feature_dim_before_pca": x.shape[1]}
    target_seeds = meta.iloc[test_ids]["seed"].to_numpy(dtype=int)
    per_seed = []
    for seed in EXPECTED["target"]:
        mask = target_seeds == seed
        raw = auroc(labels[test_ids][mask], st[mask])
        per_seed.append({"group": group, "protocol": protocol, "model": model_name,
                         "target_seed": seed, "target_auc_raw": raw,
                         "target_auc_validation_oriented": (1 - raw if flip else raw),
                         "target_auc_posthoc_oracle_diagnostic": max(raw, 1 - raw)})
    return row, per_seed


def utility_statistics(runs, baseline_csv, output):
    records = []
    for _, summary in runs:
        if summary["population"] != "target":
            continue
        best = summary.get("best") or {}
        records.append({"target_seed": int(summary["seed"]),
                        "defended_best_test_auroc": best.get("test_auroc"),
                        "defended_best_val_auroc": best.get("val_auroc"),
                        "best_round": best.get("round")})
    table = pd.DataFrame(records).sort_values("target_seed")
    stats = {"defended_best_test_auroc": bootstrap(table["defended_best_test_auroc"])}
    if baseline_csv and Path(baseline_csv).exists():
        base = pd.read_csv(baseline_csv)
        if not {"target_seed", "best_test_auroc"}.issubset(base.columns):
            raise ValueError("Baseline task CSV missing target_seed / best_test_auroc")
        if base["target_seed"].duplicated().any():
            raise ValueError("Duplicate target seeds in baseline")
        base = base[["target_seed", "best_test_auroc"]].rename(
            columns={"best_test_auroc": "baseline_best_test_auroc"})
        table = table.merge(base, on="target_seed", validate="one_to_one", how="left")
        if table["baseline_best_test_auroc"].isna().any():
            raise ValueError("Baseline missing matching target seeds")
        table["defense_minus_baseline_auroc"] = (
            table["defended_best_test_auroc"] - table["baseline_best_test_auroc"])
        stats["baseline_best_test_auroc"] = bootstrap(table["baseline_best_test_auroc"])
        stats["paired_defense_minus_baseline_auroc"] = bootstrap(
            table["defense_minus_baseline_auroc"])
        stats["paired_unit"] = "target_seed"
    else:
        stats["baseline_comparison"] = "NOT AVAILABLE; supply --baseline-task-csv"
    table.to_csv(output / "task_per_target.csv", index=False)
    (output / "task_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="All-surface HeadFix privacy audit; selects attacks only on shadow validation")
    parser.add_argument("--root", default="results/acm_revision/headfix_r150")
    parser.add_argument("--out", default="results/acm_revision/headfix_worstcase_audit")
    parser.add_argument("--groups", default=",".join(GROUPS))
    parser.add_argument("--protocols", default=",".join(PROTOCOLS))
    parser.add_argument("--models", default="LR,RF,MLP")
    parser.add_argument("--baseline-task-csv", default=(
        "results/acm_revision/core72_modality_first_v2_postprocess_r150/"
        "modality_exclusive__c0p7__r150/task_metrics_per_target_run.csv"))
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    groups = [v.strip() for v in args.groups.split(",") if v.strip()]
    protocols = [v.strip() for v in args.protocols.split(",") if v.strip()]
    models = [v.strip() for v in args.models.split(",") if v.strip()]
    if not set(groups).issubset(GROUPS) or not set(protocols).issubset(PROTOCOLS) or not set(models).issubset(("LR", "RF", "MLP")):
        raise ValueError("Unknown group/protocol/model")
    runs = discover_runs(root)
    metadata = all_metadata(runs)
    print(f"[AUDIT] verified {len(runs)} complete runs; {len(metadata)} updates", flush=True)
    utility = utility_statistics(runs, args.baseline_task_csv, out)
    rows, per_seed_rows = [], []
    total = len(groups) * len(protocols) * len(models)
    count = 0
    for group in groups:
        subset = group_metadata(metadata, group)
        if subset.empty:
            raise RuntimeError(f"No updates for surface {group}; worst-case audit incomplete")
        print(f"[AUDIT] {group}: loading {len(subset)} observed updates", flush=True)
        x = read_group_vectors(subset, group)
        y = subset["dominant_label"].to_numpy(dtype=int)
        for protocol in protocols:
            tr, va, te = get_splits(subset, protocol)
            for ids in (tr, va, te):
                if not len(ids) or len(np.unique(y[ids])) < 2:
                    raise ValueError(f"Insufficient split for {group} / {protocol}")
            for model_name in models:
                count += 1
                print(f"[AUDIT {count}/{total}] {group} | {protocol} | {model_name}", flush=True)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    row, indiv = evaluate(group, protocol, model_name, tr, va, te, y, subset, x)
                rows.append(row)
                per_seed_rows.extend(indiv)
                print(f"[RESULT] shadow-val={row['shadow_val_auc_oriented']:.4f} "
                      f"target={row['target_auc_validation_oriented']:.4f}", flush=True)
        del x
    attacks = pd.DataFrame(rows).sort_values("shadow_val_auc_oriented", ascending=False)
    attacks.to_csv(out / "all_surface_attack_grid.csv", index=False)
    by_seed = pd.DataFrame(per_seed_rows)
    by_seed.to_csv(out / "all_surface_per_target.csv", index=False)
    winner = attacks.iloc[0].to_dict()
    selected = by_seed.loc[(by_seed["group"] == winner["group"])
                           & (by_seed["protocol"] == winner["protocol"])
                           & (by_seed["model"] == winner["model"])].copy()
    selected.to_csv(out / "selected_worstcase_per_target.csv", index=False)
    report = {
        "status": "PASS",
        "rule": "Choose surface, protocol, model, and AUC orientation ONLY on shadow_val",
        "target_data_used_for_attacker_selection": False,
        "selected_group": winner["group"],
        "selected_protocol": winner["protocol"],
        "selected_model": winner["model"],
        "selection_shadow_val_oriented_auroc": winner["shadow_val_auc_oriented"],
        "target_validation_oriented_auroc": winner["target_auc_validation_oriented"],
        "target_seed_statistics": bootstrap(selected["target_auc_validation_oriented"]),
        "target_per_seed": selected[["target_seed", "target_auc_validation_oriented"]].to_dict("records"),
        "n_candidates": len(attacks),
        "groups_checked": groups, "protocols_checked": protocols, "models_checked": models,
        "utility_statistics": utility,
        "notes": [
            "The threat metric is restricted to evaluated models/protocols/surfaces.",
            "No target-label-based maximum is used to select the attack.",
            "Only observed (post-defense) updates are used by the attacker.",
            "Full-update uses the captured 2048-dimensional signed-hash representation."
        ]
    }
    (out / "selected_worstcase.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    archive_path = out.parent / (out.name + ".tar.gz")
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in out.rglob("*"):
            if path.is_file():
                archive.add(path, arcname=str(Path(out.name) / path.relative_to(out)))
    print("\n[AUDIT COMPLETE]", flush=True)
    print(f"Selected on shadow-val: {winner['group']} | {winner['protocol']} | {winner['model']}", flush=True)
    print(f"Shadow-val AUROC: {winner['shadow_val_auc_oriented']:.4f}", flush=True)
    print(f"Frozen target AUROC: {winner['target_auc_validation_oriented']:.4f}", flush=True)
    print(f"Upload result archive: {archive_path}", flush=True)


if __name__ == "__main__":
    main()
