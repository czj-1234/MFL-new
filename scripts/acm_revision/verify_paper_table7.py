#!/usr/bin/env python3
"""Reconcile manuscript Table 7 against actual per-run attack/utility audit files.

This reads independent run-level CSVs; it never synthesizes data from manuscript
values. Target-side diagnostic maxima and shadow-selected attacks are separate.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

TARGETS = (42, 43, 44, 45, 46)
METHODS = ("no_defense", "matched_only", "complete_aslwf")
EXPECTED_TABLE7 = {
    "no_defense": {"head_cpt_rf": 1.000, "all_shared_cpt_diag_max": .757,
                   "shadow_selected": 1.000, "task": .661,
                   "selected_group": "classifier_bias", "selected_protocol": "cross_partition", "selected_model": "LR"},
    "matched_only": {"head_cpt_rf": 1.000, "all_shared_cpt_diag_max": .571,
                     "shadow_selected": 1.000, "task": .658,
                     "selected_group": "classifier_head", "selected_protocol": "cross_partition", "selected_model": "LR"},
    "complete_aslwf": {"head_cpt_rf": .574, "all_shared_cpt_diag_max": .527,
                        "shadow_selected": .604, "task": .658,
                        "selected_group": "fusion", "selected_protocol": "cross_partition", "selected_model": "LR"},
}
EVALUATED_MODELS = ("LR", "RF", "MLP")
CPT = "cross_partition_temporal"


def rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def mean_five(items: list[dict], field: str) -> float:
    found = {}
    for row in items:
        seed = int(row["target_seed"])
        if seed in found:
            raise ValueError(f"duplicate target seed {seed}")
        found[seed] = float(row[field])
    if set(found) != set(TARGETS):
        raise ValueError(f"Expected five target seeds {TARGETS}; got {sorted(found)}")
    return sum(found.values()) / 5.0


def attack_score(table: list[dict], group: str, protocol: str, model: str) -> float:
    subset = [r for r in table if r.get("group") == group
              and r.get("protocol") == protocol and r.get("model") == model]
    return mean_five(subset, "target_auc_validation_oriented")


def analyze(path: Path) -> dict:
    grid = rows(path / "all_surface_per_target.csv")
    selected = rows(path / "selected_worstcase_per_target.csv")
    task = rows(path / "task_per_target.csv")
    meta_file = path / "selected_worstcase.json"
    if not meta_file.exists():
        raise FileNotFoundError(meta_file)
    metadata = json.loads(meta_file.read_text(encoding="utf-8"))
    if metadata.get("paper_milestones") is not True or tuple(metadata.get("paper_milestone_rounds") or ()) != (1, 10, 30, 50, 75, 100, 125, 150):
        raise ValueError(f"Audit has not certified eight paper milestones: {meta_file}")
    candidate_count = metadata.get("n_candidates")
    if candidate_count != 54:
        raise ValueError(f"Expected all 54 attack candidates, got {candidate_count}: {path}")
    head = attack_score(grid, "classifier_head", CPT, "RF")
    surface_max = max(attack_score(grid, "all_shared", CPT, m) for m in EVALUATED_MODELS)
    selected_mean = mean_five(selected, "target_auc_validation_oriented")
    utility = mean_five(task, "defended_best_test_auroc")
    return {
        "head_cpt_rf": head,
        "all_shared_cpt_diag_max": surface_max,
        "shadow_selected": selected_mean,
        "task": utility,
        "selected_group": metadata["selected_group"],
        "selected_protocol": metadata["selected_protocol"],
        "selected_model": metadata["selected_model"],
        "paper_milestones": metadata.get("paper_milestones"),
        "target_run_count": len(TARGETS),
        "pooled_auc_separate_diagnostic": metadata.get("target_pooled_auc_diagnostic"),
    }


def compare(observed: dict, expected: dict, tolerance: float) -> list[dict]:
    checks = []
    for key, target in expected.items():
        actual = observed[key]
        if isinstance(target, str):
            match = actual == target
        else:
            match = abs(actual - target) <= tolerance
        checks.append({"metric": key, "observed": actual, "manuscript": target,
                       "consistent": match, "difference": None if isinstance(target, str) else actual - target})
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-defense", type=Path, required=True)
    parser.add_argument("--matched-only", type=Path, required=True)
    parser.add_argument("--complete-aslwf", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=.00055,
                        help="manuscript values rounded to three decimals (default .00055)")
    parser.add_argument("--output", type=Path, default=Path("results/acm_revision/paper_table7_verification.json"))
    args = parser.parse_args()
    paths = {"no_defense": args.no_defense,
             "matched_only": args.matched_only, "complete_aslwf": args.complete_aslwf}
    report = {"status": "PASS", "scientific_unit": "five independent target FL runs",
              "metric_procedures": {
                  "head_cpt_rf": "fixed CPT/RF on classifier_head",
                  "all_shared_cpt_diag_max": "largest five-run target mean over LR/RF/MLP on all_shared (not shadow selected)",
                  "shadow_selected": "fixed shadow-val-selected configuration and orientation",
                  "task": "mean test AUROC at each run's independently validation-selected checkpoint",
              },
              "analyses": {}, "uncertified": []}
    for method, path in paths.items():
        try:
            actual = analyze(path)
            checks = compare(actual, EXPECTED_TABLE7[method], args.tolerance)
            report["analyses"][method] = {"source": str(path), "observed": actual, "checks": checks}
            if not all(x["consistent"] for x in checks):
                report["status"] = "FAIL"
        except Exception as exc:
            report["status"] = "FAIL"
            report["analyses"][method] = {"source": str(path), "error": f"{type(exc).__name__}: {exc}"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(args.output),
                      "methods": list(report["analyses"])}, ensure_ascii=False))
    return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
