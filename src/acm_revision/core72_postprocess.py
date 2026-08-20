from __future__ import annotations

import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

from .attacks import AttackSplit, make_attack_split, read_metadata, run_attack_grid, run_tabular_attack
from .privacy_postprocess import (
    ALL_GROUPS, _available, canonical_leakage_probabilities,
    label_distribution_inference, run_trajectory_suite, structural_privacy,
)

DEFAULT_SEEDS = {"shadow_train": [142, 143], "shadow_val": [242], "target": [42, 43, 44, 45, 46]}
POOLED_MODELS = ["random", "majority", "update_norm", "cosine_centroid", "logistic_regression", "linear_svm", "random_forest", "boosted_trees", "mlp"]
POOLED_PROTOCOLS = ["random_update", "temporal", "cross_client", "cross_partition", "cross_partition_temporal"]


def discover(root, setting, concentration, rounds, seeds):
    out = []
    allowed = {k: set(map(int, v)) for k, v in seeds.items()}
    for p in Path(root).rglob("update_metadata.csv"):
        s, m = p.parent / "summary.json", p.parent / "capture_manifest.json"
        if not s.exists() or not m.exists():
            continue
        try:
            summary, manifest = json.loads(s.read_text()), json.loads(m.read_text())
        except Exception:
            continue
        pop = str(summary.get("population"))
        if pop not in allowed or manifest.get("status") != "PASS":
            continue
        if str(summary.get("setting_name")) != setting or int(summary.get("rounds", -1)) != rounds:
            continue
        if int(summary.get("seed", -1)) not in allowed[pop]:
            continue
        if not math.isclose(float(summary.get("concentration")), concentration, abs_tol=1e-9):
            continue
        out.append(p)
    return sorted(out)


def validate_matrix(frame, setting, concentration, seeds):
    failures = []
    if set(frame.setting_name.astype(str).unique()) != {setting}:
        failures.append("mixed settings")
    for pop, expected in seeds.items():
        got = sorted(map(int, frame.loc[frame.population == pop, "seed"].unique()))
        if got != sorted(map(int, expected)):
            failures.append(f"{pop}: expected {expected}, got {got}")
    runs = frame[["population", "seed", "run_id"]].drop_duplicates()
    if len(runs) != sum(map(len, seeds.values())):
        failures.append(f"expected {sum(map(len, seeds.values()))} runs, got {len(runs)}")
    if failures:
        raise RuntimeError("; ".join(failures))
    return {"status": "PASS", "setting": setting, "concentration": concentration, "n_runs": len(runs), "n_updates": len(frame)}


def bootstrap(values):
    x = np.asarray([v for v in values if pd.notna(v)], float)
    if len(x) < 2:
        return {"mean": float(x[0]) if len(x) else np.nan, "std": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": len(x)}
    rng = np.random.default_rng(42)
    b = rng.choice(x, (5000, len(x)), replace=True).mean(1)
    return {"mean": x.mean(), "std": x.std(ddof=1), "ci_low": np.quantile(b, .025), "ci_high": np.quantile(b, .975), "n": len(x)}


def strict_per_target(frame, group, target_seeds, out_dir, smoke=False):
    rows = []
    protocols = ["cross_partition"] if smoke else ["cross_partition", "cross_partition_temporal"]
    models = ["logistic_regression"] if smoke else ["logistic_regression", "random_forest", "mlp"]
    subset = _available(frame, group)
    for protocol in protocols:
        base = make_attack_split(subset, protocol, seed=42)
        for seed in target_seeds:
            test = base.test[base.test.seed.astype(int) == int(seed)]
            split = AttackSplit(base.train, base.val, test, protocol)
            for model in models:
                try:
                    r = run_tabular_attack(split, group, model, "dominant_label", "observed", 42, 256)
                    r["target_seed"] = int(seed)
                    rows.append(r)
                except Exception as exc:
                    rows.append({"protocol": protocol, "group": group, "attack_model": model, "target": "dominant_label", "target_seed": int(seed), "error": repr(exc)})
    per = pd.DataFrame(rows)
    per.to_csv(out_dir / f"strict_independent_runs__{group}.csv", index=False)
    ok = per if "error" not in per else per[per.error.isna() | (per.error.astype(str).str.strip() == "")]
    agg = []
    for keys, part in ok.groupby(["protocol", "group", "attack_model", "target"], dropna=False):
        row = dict(zip(["protocol", "group", "attack_model", "target"], keys))
        row["independent_unit"] = "target_fl_seed"
        for metric in ["attack_acc", "attack_asr", "attack_macro_f1", "attack_balanced_acc", "attack_auroc"]:
            for suffix, value in bootstrap(part[metric]).items():
                row[f"{metric}_{suffix}"] = value
        agg.append(row)
    pd.DataFrame(agg).to_csv(out_dir / f"strict_independent_summary__{group}.csv", index=False)


def task_rows(paths, out_dir):
    rows = []
    for p in paths:
        summary = json.loads((p.parent / "summary.json").read_text())
        if summary.get("population") != "target":
            continue
        best = summary.get("best") or {}
        row = {"run_id": summary.get("run_id"), "setting_name": summary.get("setting_name"), "concentration": summary.get("concentration"), "target_seed": int(summary["seed"]), "rounds": int(summary["rounds"]), "best_round": best.get("round")}
        row.update({f"best_{k}": v for k, v in best.items() if k != "run_id"})
        rp = p.parent / "round_metrics.csv"
        if rp.exists():
            final = pd.read_csv(rp).sort_values("round").iloc[-1].to_dict()
            row.update({f"final_{k}": v for k, v in final.items()})
        rows.append(row)
    result = pd.DataFrame(rows).sort_values("target_seed")
    result.to_csv(out_dir / "task_metrics_per_target_run.csv", index=False)
    return result


def run(root, setting, concentration, rounds, output_root, seeds, smoke=False):
    paths = discover(root, setting, concentration, rounds, seeds)
    if not paths:
        raise FileNotFoundError("no matching validated runs")
    frame = read_metadata(paths)
    matrix = validate_matrix(frame, setting, concentration, seeds)
    out = Path(output_root) / f"{setting}__c{str(concentration).replace('.', 'p')}__r{rounds}"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "combined_update_metadata.csv", index=False)
    tasks = task_rows(paths, out)
    groups = ["classifier_head", "fusion", "full_update"] if smoke else ALL_GROUPS
    canonical, label = [], []
    for group in groups:
        gf = _available(frame, group)
        if gf.empty:
            continue
        gm = out / f"metadata__{group}.csv"
        gf.to_csv(gm, index=False)
        run_attack_grid([gm], out / f"attack_summary__{group}.csv", ["cross_partition"] if smoke else POOLED_PROTOCOLS, [group], ["logistic_regression"] if smoke else POOLED_MODELS, ["dominant_label"] if smoke else ["dominant_label", "modality", "round", "random_group"], "observed", 42, 256)
        if not smoke:
            run_attack_grid([gm], out / f"identity_control__{group}.csv", ["random_update", "temporal"], [group], ["logistic_regression", "random_forest", "mlp"], ["client_id"], "observed", 42, 256)
        strict_per_target(frame, group, seeds["target"], out, smoke)
        try:
            r, _ = canonical_leakage_probabilities(frame, group, out)
            r["interpretation"] = "nominal_role_separability_attribution_control" if concentration == .5 else "dominant_label_leakage"
            canonical.append(r)
            lp = out / f"leakage_probabilities__{group}.csv"
            q = pd.read_csv(lp); q["interpretation"] = r["interpretation"]; q.to_csv(lp, index=False)
        except Exception as exc:
            canonical.append({"group": group, "error": repr(exc)})
        try:
            r = label_distribution_inference(frame, group, out)
            r["interpretation"] = "balanced_distribution_control" if concentration == .5 else "label_distribution_inference"
            label.append(r)
        except Exception as exc:
            label.append({"group": group, "error": repr(exc)})
    pd.DataFrame(canonical).to_csv(out / "canonical_leakage_summary.csv", index=False)
    pd.DataFrame(label).to_csv(out / "label_distribution_summary.csv", index=False)
    structural = structural_privacy(frame, groups, out / "structural_privacy.csv")
    trajectory = run_trajectory_suite(frame, out, smoke)
    report = {"status": "PASS", "setting": setting, "concentration": concentration, "rounds": rounds, "balanced_attribution_control": concentration == .5, "matrix_validation": matrix, "output_dir": str(out), "n_target_task_runs": len(tasks), "n_structural_rows": len(structural), "n_trajectory_rows": len(trajectory), "representation_note": "head/fusion exact; encoders/all_shared coordinate sketches; full_update signed feature-hash projection using every coordinate"}
    (out / "postprocess_report.json").write_text(json.dumps(report, indent=2))
    return report


def parse_ints(s):
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/acm_revision/core72_r150")
    p.add_argument("--setting", required=True, choices=["image_only", "text_only", "modality_exclusive"])
    p.add_argument("--concentration", type=float, required=True, choices=[.5, .7, .9])
    p.add_argument("--rounds", type=int, default=150)
    p.add_argument("--output-root", default="results/acm_revision/core72_postprocess_r150")
    p.add_argument("--shadow-train-seeds", default="142,143")
    p.add_argument("--shadow-val-seeds", default="242")
    p.add_argument("--target-seeds", default="42,43,44,45,46")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    seeds = {"shadow_train": parse_ints(a.shadow_train_seeds), "shadow_val": parse_ints(a.shadow_val_seeds), "target": parse_ints(a.target_seeds)}
    print(json.dumps(run(a.root, a.setting, a.concentration, a.rounds, a.output_root, seeds, a.smoke), indent=2))


if __name__ == "__main__":
    main()
