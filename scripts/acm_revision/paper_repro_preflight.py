#!/usr/bin/env python3
"""Read-only provenance/preflight for the 2026-10-17 ASLWF manuscript.

Preflight checks are not an independent numerical reproduction. They verify
protocol invariants and report missing original artifacts without fabricating
results. Exit code is nonzero for a FAIL or for any missing artifact in strict mode.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PAPER_ROUNDS = (1, 10, 30, 50, 75, 100, 125, 150)
EXPECTED_TARGET_SEEDS = (42, 43, 44, 45, 46)
EXPECTED_GROUPS = (
    "classifier_bias", "classifier_weight", "classifier_head", "fusion",
    "missing_modality", "image_encoder", "text_encoder", "all_shared", "full_update",
)
SOURCE_FILES = (
    "src/acm_revision/core72_modality_first_plan.py",
    "src/acm_revision/defense70_prepare.py",
    "src/acm_revision/defense70_plan.py",
    "src/acm_revision/headfix_prepare.py",
    "src/acm_revision/headfix_plan.py",
    "src/acm_revision/paper_protocol_plan.py",
    "src/acm_revision/defense_dispatch.py",
    "src/acm_revision/hash_subspace.py",
    "scripts/acm_revision/audit_headfix.py",
    "docs/paper_repro/README.md",
    "docs/paper_repro/TRACEABILITY.csv",
)
OLD_MANIFEST = "results/acm_revision/defense70_prep/locked_params.yaml"
HEAD_MANIFEST = "results/acm_revision/headfix_prep/locked_headfix.yaml"
HEAD_RUN_ROOT = "results/acm_revision/headfix_r150"


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to check locked manifests") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping: {path}")
    return data


def run(root: Path, strict: bool = False) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []

    def add(name: str, status: str, detail: str) -> None:
        findings.append({"check": name, "status": status, "detail": detail})

    for rel in SOURCE_FILES:
        add("source:" + rel, "PASS" if (root / rel).is_file() else "FAIL",
            "present" if (root / rel).is_file() else "missing source file")

    head_src = (root / "src/acm_revision/headfix_prepare.py")
    plan_src = (root / "src/acm_revision/headfix_plan.py")
    audit_src = (root / "scripts/acm_revision/audit_headfix.py")
    if head_src.exists():
        t = head_src.read_text(encoding="utf-8")
        predicates = ("TRAIN_SEEDS = (142, 143)", "VAL_SEEDS = (242,)",
                      "TRAIN_MAX_ROUND = 60", "VAL_MIN_ROUND = 61",
                      "_build_iterative_basis", "worst_oriented_auroc")
        add("head-basis-shadow-only-code", "PASS" if all(x in t for x in predicates) else "FAIL",
            "shadow seeds, temporal train/selection split and iterative LR selection")
    if plan_src.exists():
        t = plan_src.read_text(encoding="utf-8")
        patterns = ('"name": "layerwise_mixed_filter"',
                    '_exact_filter("classifier_head", head_basis',
                    '_exact_filter("fusion", old_basis',
                    '_exact_filter("missing_modality", old_basis',
                    '_hash_filter("image_encoder", old_basis',
                    '_hash_filter("text_encoder", old_basis',
                    '"all_defense_parameters_frozen_before_formal_target_runs": True')
        add("combined-five-group-filter-implementation",
            "PASS" if all(x in t for x in patterns) else "FAIL",
            "attacker-guided head replaces matched head; other four groups remain matched-only")
        capture10 = "milestones = [1, 5, 10, 20, 30, 50, 75, 100, 125, 150]" in t
        add("historical-headfix-capture-calendar",
            "INFO" if capture10 else "WARN",
            "headfix captures 10 milestones; paper attacks must select canonical eight milestones")
        add("headfix-full-hash-2048",
            "PASS" if '"full_group_projection_dim": 2048' in t else "FAIL",
            "paper full-update observation must be a 2048-dimensional all-coordinate signed hash")

    if audit_src.exists():
        t = audit_src.read_text(encoding="utf-8")
        predicates = ("--paper-milestones", "PAPER_MILESTONE_ROUNDS", "paper_milestones")
        add("eight-milestone-paper-audit-available",
            "PASS" if all(p in t for p in predicates) else "FAIL",
            "paper-aligned attack filtering must be explicitly requested; legacy audit remains available")

    # Historical conflict is recorded, never hidden by modifying old configs.
    d70plan = root / "src/acm_revision/defense70_plan.py"
    if d70plan.exists():
        old = d70plan.read_text(encoding="utf-8")
        add("legacy-defense70-full-update-hash",
            "WARN" if '"full_group_projection_dim": 8192' in old else "INFO",
            "archive uses 8192 rather than paper's 2048; cannot claim its historical full-update attack is paper-equivalent without new consistent processing")

    artifacts: dict[str, dict[str, Any]] = {}
    for name, rel in (("matched", OLD_MANIFEST), ("head", HEAD_MANIFEST)):
        path = root / rel
        if not path.is_file():
            add(name + "-locked-manifest", "MISSING", f"not found: {rel}")
            continue
        try:
            manifest = read_yaml(path)
            artifacts[name] = manifest
            if manifest.get("status") != "LOCKED":
                add(name + "-locked-manifest", "FAIL", "status is not LOCKED")
            elif manifest.get("provenance", {}).get("target_data_used_for_basis_or_tuning") is not False:
                add(name + "-locked-manifest", "FAIL", "target-free provenance not certified")
            else:
                add(name + "-locked-manifest", "PASS", "locked and target-free as recorded")
            if name == "matched":
                point = manifest.get("selected", {}).get("strong", {})
                okay = int(point.get("rank", -1)) == 5 and abs(float(point.get("alpha", -1)) - 1.0) < 1e-8
                add("matched-strong-operating-point", "PASS" if okay else "FAIL",
                    "expected r=5, alpha=1; actual=" + json.dumps(point, default=str))
            if name == "head":
                sel = manifest.get("selected", {})
                rank = sel.get("rank")
                okay = isinstance(rank, int) and rank > 0 and abs(float(sel.get("alpha", -1)) - 1) < 1e-8
                add("actual-selected-head-rank", "PASS" if okay else "FAIL",
                    "selected rank=" + str(rank) + "; MAX_RANK=64 is not the selected rank")
                prv = manifest.get("provenance", {})
                source_ok = list(prv.get("basis_seeds", [])) == [142, 143] and list(prv.get("validation_seeds", [])) == [242]
                add("head-basis-populations", "PASS" if source_ok else "FAIL",
                    "shadow train seeds 142,143; shadow validation 242")
        except Exception as exc:
            add(name + "-locked-manifest", "FAIL", f"{type(exc).__name__}: {exc}")

    if artifacts.get("matched") and artifacts.get("head"):
        add("separate-basis-manifests", "PASS", "both separately selected sources exist")

    rroot = root / HEAD_RUN_ROOT
    if not rroot.is_dir():
        add("headfix-formal-runs", "MISSING", f"not found: {HEAD_RUN_ROOT}")
    else:
        discovered: dict[tuple[str, int], list[str]] = {}
        for summary_path in rroot.rglob("summary.json"):
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                p = str(summary.get("population", ""))
                s = int(summary.get("seed", -1))
                if p not in ("shadow_train", "shadow_val", "target"):
                    continue
                discovered.setdefault((p, s), []).append(str(summary_path.parent.relative_to(root)))
                if int(summary.get("rounds", 0)) != 150:
                    add("incomplete-fl:" + p + ":" + str(s), "FAIL", "not 150 rounds")
                capture = summary_path.parent / "capture_manifest.json"
                if not capture.exists() or json.loads(capture.read_text(encoding="utf-8")).get("status") != "PASS":
                    add("capture:" + p + ":" + str(s), "FAIL", "capture manifest missing or not PASS")
                if p == "target" and (summary.get("best") or {}).get("test_auroc") is None:
                    add("utility:" + str(s), "FAIL", "missing independently trained target best test AUROC")
            except Exception as exc:
                add("headfix-run-parse:" + str(summary_path), "FAIL", str(exc))
        expected = {("shadow_train", 142), ("shadow_train", 143),
                    ("shadow_val", 242)} | {("target", s) for s in EXPECTED_TARGET_SEEDS}
        got = set(discovered)
        if expected != got or any(len(v) != 1 for v in discovered.values()):
            add("headfix-formal-runs", "FAIL",
                "expected 8 unique population/seed runs; missing="
                + str(sorted(expected - got)) + "; extra=" + str(sorted(got - expected)))
        else:
            add("headfix-formal-runs", "PASS", "8 distinct 150-round runs present; target seeds 42–46")
        if not discovered:
            add("headfix-formal-runs", "FAIL", "no valid population/seed summaries")

    failed = sum(f["status"] == "FAIL" for f in findings)
    missing = sum(f["status"] == "MISSING" for f in findings)
    warnings = sum(f["status"] == "WARN" for f in findings)
    outcome = "FAIL" if failed or (strict and missing) else ("NEEDS_ARTIFACTS" if missing else ("REVIEW_REQUIRED" if warnings else "PASS"))
    return {"status": outcome, "strict_artifacts": strict,
            "paper_attack_milestones": list(PAPER_ROUNDS),
            "paper_attack_configurations": len(EXPECTED_GROUPS) * 3 * 2,
            "counts": {"fail": failed, "missing": missing, "warnings": warnings},
            "findings": findings}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--strict-artifacts", action="store_true", help="missing original artifact is a failing condition")
    p.add_argument("--json-out", type=Path, help="optional JSON report file (never overwrites model outputs)")
    args = p.parse_args()
    report = run(args.root.resolve(), args.strict_artifacts)
    formatted = json.dumps(report, indent=2, ensure_ascii=False)
    print(formatted)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(formatted + "\n", encoding="utf-8")
    return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
