from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

SETTINGS = ("image_only", "text_only", "modality_exclusive")
CONCENTRATIONS = (0.5, 0.7, 0.9)
TARGET_SEEDS = (42, 43, 44, 45, 46)


def find_cells(roots):
    found = {}
    for root in map(Path, roots):
        for report_path in root.rglob("postprocess_report.json"):
            try:
                report = json.loads(report_path.read_text())
            except Exception:
                continue
            key = (report.get("setting"), float(report.get("concentration", -1)))
            if key[0] in SETTINGS and key[1] in CONCENTRATIONS:
                if key in found and found[key] != report_path.parent:
                    raise RuntimeError(f"duplicate cell {key}")
                found[key] = report_path.parent
    expected = {(s, c) for s in SETTINGS for c in CONCENTRATIONS}
    missing = sorted(expected - set(found))
    if missing:
        raise RuntimeError(f"missing Core72 cells: {missing}")
    return [found[(s, c)] for s in SETTINGS for c in CONCENTRATIONS]


def successful(frame):
    if "error" not in frame:
        return frame
    return frame[frame.error.isna() | (frame.error.astype(str).str.strip() == "")]


def bootstrap(values):
    x = np.asarray([v for v in values if pd.notna(v)], float)
    if len(x) < 2:
        return {"mean": float(x[0]) if len(x) else np.nan, "std": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": len(x)}
    rng = np.random.default_rng(42)
    b = rng.choice(x, (5000, len(x)), replace=True).mean(1)
    return {"mean": x.mean(), "std": x.std(ddof=1), "ci_low": np.quantile(b, .025), "ci_high": np.quantile(b, .975), "n": len(x)}


def aggregate(frame, group_cols, metrics):
    rows = []
    for keys, part in frame.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        for metric in metrics:
            if metric in part:
                for suffix, value in bootstrap(part[metric]).items():
                    row[f"{metric}_{suffix}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def paired(frame, metric, fixed_cols):
    rows = []
    for baseline in ("image_only", "text_only"):
        left = frame[frame.setting_name == "modality_exclusive"]
        right = frame[frame.setting_name == baseline]
        left = left[fixed_cols + [metric]].rename(columns={metric: "left"})
        right = right[fixed_cols + [metric]].rename(columns={metric: "right"})
        merged = left.merge(right, on=fixed_cols).dropna()
        if len(merged) < 2:
            continue
        x, y = merged.left.to_numpy(float), merged.right.to_numpy(float)
        t = ttest_rel(x, y)
        try:
            w = wilcoxon(x, y)
            ws, wp = float(w.statistic), float(w.pvalue)
        except Exception:
            ws, wp = np.nan, np.nan
        row = {
            "comparison": f"modality_exclusive_vs_{baseline}",
            "metric": metric,
            "n_pairs": len(merged),
            "mean_paired_difference": float((x - y).mean()),
            "paired_t_statistic": float(t.statistic),
            "paired_t_pvalue": float(t.pvalue),
            "wilcoxon_statistic": ws,
            "wilcoxon_pvalue": wp,
        }
        row.update({k: merged.iloc[0][k] for k in fixed_cols if k != "target_seed"})
        rows.append(row)
    return pd.DataFrame(rows)


def run(roots: Sequence[str | Path], output_dir: str | Path):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cells = find_cells(roots)
    task_frames, attack_frames = [], []
    for cell in cells:
        report = json.loads((cell / "postprocess_report.json").read_text())
        setting, concentration = report["setting"], float(report["concentration"])
        task = pd.read_csv(cell / "task_metrics_per_target_run.csv")
        task["setting_name"], task["concentration"] = setting, concentration
        task_frames.append(task)
        for path in cell.glob("strict_independent_runs__*.csv"):
            attack = successful(pd.read_csv(path))
            attack["setting_name"], attack["concentration"] = setting, concentration
            attack_frames.append(attack)
    tasks = pd.concat(task_frames, ignore_index=True)
    attacks = pd.concat(attack_frames, ignore_index=True)
    tasks.to_csv(out / "core72_task_per_target_run.csv", index=False)
    attacks.to_csv(out / "core72_attack_per_target_run.csv", index=False)

    task_metrics = [x for x in ["best_test_acc", "best_test_macro_f1", "best_test_auroc", "final_test_acc", "final_test_macro_f1", "final_test_auroc"] if x in tasks]
    attack_metrics = [x for x in ["attack_acc", "attack_asr", "attack_macro_f1", "attack_balanced_acc", "attack_auroc"] if x in attacks]
    aggregate(tasks, ["setting_name", "concentration"], task_metrics).to_csv(out / "core72_task_summary.csv", index=False)
    aggregate(attacks, ["setting_name", "concentration", "protocol", "group", "attack_model", "target"], attack_metrics).to_csv(out / "core72_attack_summary.csv", index=False)

    task_tests = []
    for c in CONCENTRATIONS:
        part = tasks[tasks.concentration == c]
        for metric in task_metrics:
            x = paired(part, metric, ["concentration", "target_seed"])
            if not x.empty:
                task_tests.append(x)
    pd.concat(task_tests, ignore_index=True).to_csv(out / "core72_task_paired_tests.csv", index=False)

    attack_tests = []
    keys = ["concentration", "protocol", "group", "attack_model", "target"]
    for _, part in attacks.groupby(keys, dropna=False):
        for metric in attack_metrics:
            x = paired(part, metric, keys + ["target_seed"])
            if not x.empty:
                attack_tests.append(x)
    pd.concat(attack_tests, ignore_index=True).to_csv(out / "core72_attack_paired_tests.csv", index=False)

    failures = []
    expected_rows = len(SETTINGS) * len(CONCENTRATIONS) * len(TARGET_SEEDS)
    if len(tasks) != expected_rows:
        failures.append(f"expected {expected_rows} task rows, got {len(tasks)}")
    report = {
        "status": "PASS" if not failures else "FAIL",
        "roots": list(map(str, roots)),
        "cell_dirs": list(map(str, cells)),
        "n_task_rows": len(tasks),
        "n_attack_rows": len(attacks),
        "independent_unit": "target_fl_seed",
        "balanced_control_note": "Concentration 0.5 is an attribution control, not a privacy defense.",
        "failures": failures,
    }
    (out / "core72_aggregate_report.json").write_text(json.dumps(report, indent=2))
    if failures:
        raise RuntimeError("; ".join(failures))
    (out / "CORE72_AGGREGATE_PASS").write_text("PASS\n")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--post-roots", nargs="+", required=True)
    p.add_argument("--output-dir", default="results/acm_revision/core72_final")
    a = p.parse_args()
    print(json.dumps(run(a.post_roots, a.output_dir), indent=2))


if __name__ == "__main__":
    main()
