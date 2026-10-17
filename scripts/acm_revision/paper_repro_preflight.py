#!/usr/bin/env python3
"""Strict single-protocol source and artifact preflight for the 2026-10-17 paper.

Exactly 8 milestone rounds, 2048-d full-update signed hash, and enforced
original-coordinate common L2 budgets. No legacy/alternate protocol modes.
A source PASS is not numerical reproduction; real artifacts must be verified.
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
    "src/acm_revision/core72_plan.py",
    "src/acm_revision/defense70_prepare.py",
    "src/acm_revision/defense70_plan.py",
    "src/acm_revision/headfix_prepare.py",
    "src/acm_revision/headfix_plan.py",
    "src/acm_revision/defense_dispatch.py",
    "src/acm_revision/hash_subspace.py",
    "src/acm_revision/l2_budget.py",
    "src/acm_revision/paper_protocol_plan.py",
    "src/acm_revision/privacy_capture_runner_v2.py",
    "scripts/acm_revision/audit_headfix.py",
    "scripts/acm_revision/verify_paper_table7.py",
    "docs/paper_repro/README.md",
    "docs/paper_repro/TRACEABILITY.csv",
)
OLD_MANIFEST = "results/acm_revision/defense70_prep/locked_params.yaml"
HEAD_MANIFEST = "results/acm_revision/headfix_prep/locked_headfix.yaml"
HEAD_RUN_ROOT = "results/acm_revision/paper_repro_20261017/complete"


def _yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML required for locked manifests") from exc
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("Expected mapping: " + str(path))
    return obj


def run(root: Path, strict: bool = False) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []

    def add(name: str, status: str, detail: str) -> None:
        findings.append({"check": name, "status": status, "detail": detail})

    for path in SOURCE_FILES:
        add("source:" + path, "PASS" if (root / path).is_file() else "FAIL", path)

    def check_code(path: str, substrings: list[str], name: str, excluded: list[str] | None = None) -> None:
        p = root / path
        if not p.is_file():
            return
        text = p.read_text(encoding="utf-8")
        matches = all(token in text for token in substrings)
        incompatible = [s for s in (excluded or ()) if s in text]
        add(name, "PASS" if matches and not incompatible else "FAIL",
            "Required code present; incompatible markers=" + str(incompatible))

    check_code("src/acm_revision/paper_protocol_plan.py",
               ["PAPER_MILESTONES = (1, 10, 30, 50, 75, 100, 125, 150)",
                "PAPER_FULL_UPDATE_DIM = 2048", "def _core_templates(",
                'if name == "matched":', '"core", "baselines", "complete"'],
               "single-canonical-training-config")
    check_code("src/acm_revision/defense70_plan.py",
               ['"full_group_projection_dim": 2048',
                '"max_relative_update_distortion_l2"', "weak_l2", "strong_l2"],
               "same-protocol-all-baselines",
               excluded=['"full_group_projection_dim": 8192'])
    check_code("src/acm_revision/headfix_plan.py",
               ["milestones = [1, 10, 30, 50, 75, 100, 125, 150]",
                '"max_relative_update_distortion_l2": strong_l2',
                '"name": "layerwise_mixed_filter"'],
               "same-protocol-complete-aslwf",
               excluded=["milestones = [1, 5, 10, 20, 30, 50, 75, 100, 125, 150]"])
    check_code("src/acm_revision/privacy_capture_runner.py",
               ["DEFAULT_MILESTONE_ROUNDS = [1, 10, 30, 50, 75, 100, 125, 150]"],
               "default-capture-eight")
    check_code("src/acm_revision/privacy_capture_runner_v2.py",
               ["if projection_dim != 2048:", "Paper protocol requires exactly eight fixed capture rounds"],
               "strict-2048-capture")
    check_code("scripts/acm_revision/audit_headfix.py",
               ["PAPER_MILESTONE_ROUNDS", 'metadata["round"].astype(int).isin(PAPER_MILESTONE_ROUNDS)',
                "milestone_rounds_present", "target_mean_per_independent_run_auroc"],
               "strict-eight-round-defense-aware-audit",
               excluded=["parser.add_argument(\"--paper-milestones\""])
    check_code("src/acm_revision/defense70_prepare.py",
               ['"l2_budget": {', '"calibration_population": "shadow_val"',
                '"weak_max_relative_update_distortion_l2"',
                '"strong_max_relative_update_distortion_l2"'],
               "shadow-locked-common-l2-budget")
    check_code("src/acm_revision/defense_dispatch.py",
               ["apply_common_l2_cap_inplace", "def _enforce_shared_l2(",
                'detail = _enforce_shared_l2', "summary = _enforce_shared_l2"],
               "actual-before-upload-l2-enforcement")
    check_code("src/acm_revision/l2_budget.py",
               ['"budget_verified": within_budget', "raw_norm", "distortion_sq",
                "num_original_floating_coordinates"],
               "measure-original-update-l2-not-sketch")

    # Only the canonical generated directory is permitted after job generation.
    generated = root / "configs/acm_revision/generated"
    extra = []
    canonical = (generated / "paper_repro_20261017").resolve()
    if generated.exists():
        for item in generated.rglob("*"):
            if item.is_file() and not item.resolve().is_relative_to(canonical):
                extra.append(str(item.relative_to(root)))
    add("one-generated-config-root", "FAIL" if extra else "PASS",
        "Non-canonical generated files: " + str(extra[:10]))

    matched = root / OLD_MANIFEST
    head = root / HEAD_MANIFEST
    for name, path in (("matched", matched), ("head", head)):
        if not path.is_file():
            add(name + "-locked-manifest", "MISSING", "Real original lock not supplied: " + str(path))
            continue
        try:
            doc = _yaml(path)
            ok = doc.get("status") == "LOCKED" and doc.get("provenance", {}).get(
                "target_data_used_for_basis_or_tuning") is False
            add(name + "-locked-manifest", "PASS" if ok else "FAIL", "must be LOCKED and target-free")
            if name == "matched":
                l2 = doc.get("l2_budget", {})
                good = l2.get("target_data_used") is False
                for tier in ("weak", "strong"):
                    x = float(l2[tier + "_max_relative_update_distortion_l2"])
                    good = good and 0.0 < x <= 1.0
                add("matched-common-l2-budget", "PASS" if good else "FAIL",
                    "Both tiers must have the same locked cap per compared method")
                w, s = doc["selected"]["weak"], doc["selected"]["strong"]
                pts = (int(w["rank"]) == 1 and abs(float(w["alpha"]) - .25) < 1e-9
                       and int(s["rank"]) == 5 and abs(float(s["alpha"]) - 1) < 1e-9)
                add("matched-selected-weak-strong", "PASS" if pts else "FAIL",
                    "Expected (r,alpha)=(1,.25),(5,1) if reproducing manuscript")
            else:
                rank = int(doc.get("selected", {}).get("rank", 0))
                alpha = float(doc.get("selected", {}).get("alpha", -1))
                add("head-final-selected-rank", "PASS" if rank > 0 and abs(alpha - 1) < 1e-9 else "FAIL",
                    "Actual locked rank=" + str(rank) + ", not merely MAX_RANK")
        except Exception as exc:
            add(name + "-locked-manifest", "FAIL", str(exc))

    root_runs = root / HEAD_RUN_ROOT
    if not root_runs.exists():
        add("complete-formal-fl-runs", "MISSING", "Eight independent defended run outputs unavailable")
    else:
        found: dict[tuple[str, int], list[Path]] = {}
        expected = {("shadow_train", 142), ("shadow_train", 143), ("shadow_val", 242)}
        expected.update(("target", s) for s in EXPECTED_TARGET_SEEDS)
        for path in root_runs.rglob("summary.json"):
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))
                population = str(summary.get("population", ""))
                seed = int(summary.get("seed", -1))
                if population not in ("shadow_train", "shadow_val", "target"):
                    continue
                found.setdefault((population, seed), []).append(path)
                capture_path = path.parent / "capture_manifest.json"
                capture = json.loads(capture_path.read_text(encoding="utf-8"))
                rounds = tuple(sorted(int(x) for x in capture.get("milestone_rounds_present", [])))
                passed = (int(summary.get("rounds", 0)) == 150
                          and capture.get("status") == "PASS" and rounds == PAPER_ROUNDS)
                add("run:" + population + ":" + str(seed), "PASS" if passed else "FAIL",
                    "150 rounds; capture PASS; exact eight milestones")
                if population == "target" and (summary.get("best") or {}).get("test_auroc") is None:
                    add("target-utility:" + str(seed), "FAIL", "Missing independent best test AUROC")
            except Exception as exc:
                add("run-parse:" + str(path), "FAIL", str(exc))
        good = set(found) == expected and all(len(v) == 1 for v in found.values())
        add("complete-formal-fl-runs", "PASS" if good else "FAIL",
            "Need exactly eight independent runs: expected=" + str(sorted(expected))
            + "; obtained=" + str(sorted(found)))

    failures = sum(x["status"] == "FAIL" for x in findings)
    missing = sum(x["status"] == "MISSING" for x in findings)
    status = "FAIL" if failures or (strict and missing) else ("NEEDS_ARTIFACTS" if missing else "PASS")
    return {"status": status, "strict_artifacts": strict,
            "protocol": {"milestone_rounds": list(PAPER_ROUNDS),
                         "full_update_dim": 2048,
                         "attacks": len(EXPECTED_GROUPS) * 3 * 2,
                         "l2_definition": "||Delta_filtered-Delta_raw||2 <= rho*||Delta_raw||2"},
            "counts": {"fail": failures, "missing": missing}, "findings": findings}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--strict-artifacts", action="store_true")
    p.add_argument("--json-out", type=Path)
    args = p.parse_args()
    output = run(args.root.resolve(), args.strict_artifacts)
    report = json.dumps(output, ensure_ascii=False, indent=2)
    print(report)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(report + "\n", encoding="utf-8")
    return 1 if output["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
