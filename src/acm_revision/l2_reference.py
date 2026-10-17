"""Read-only paper L2 reference thresholds derived from l2_summary(1).xlsx.

NEVER alters training updates. Source contains 12 aggregate MEANS for one
shadow-validation run (n_updates=96 per method/surface), not per-update maxima.
'full_update' data are PROXY_ONLY 2048-d signed-hash observations and MUST NOT
be used as a full-original-update distortion cap.

Use these thresholds to compare already-calculated group/tier MEANS without
retraining. This cannot establish that a hard L2 cap existed during prior FL.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

SOURCE_XLSX_SHA256 = "ce3e4411714879eae23123631bbeaac2fdd056f6b061c28e7a293f98ac9e91a3"
SOURCE_FILENAME = "l2_summary(1).xlsx"
REFERENCE_CSV = (
    Path(__file__).resolve().parents[2]
    / "docs" / "paper_repro" / "l2_summary_reference.csv"
)
# The largest REPORTED mean_relative_L2 in each method-tier/surface cell,
# not the maximum over individual clients/rounds.
MEAN_RELATIVE_L2_THRESHOLDS = {
    "strong": {
        "classifier_head": 0.8175672136188386,
        "full_update": 0.014964902194092806,
    },
    "weak": {
        "classifier_head": 0.20400139029459616,
        "full_update": 0.0010105610364926128,
    },
}
SURFACE_MEASUREMENT = {
    "classifier_head": "EXACT_SURFACE",
    "full_update": "PROXY_ONLY",
}
ALLOWED_METHODS = {"pca", "random", "proposed_system"}


def method_tier(method: str) -> tuple[str, str]:
    for tier in ("strong", "weak"):
        suffix = "_" + tier
        if method.endswith(suffix):
            family = method[:-len(suffix)]
            if family in ALLOWED_METHODS:
                return family, tier
    raise ValueError("Unknown method/tier in original L2 summary: " + method)


def threshold(surface: str, tier: str) -> float:
    """Get published-summary reference; this is NOT a live training cap."""
    return MEAN_RELATIVE_L2_THRESHOLDS[tier][surface]


def audit_existing_means(source: str | Path = REFERENCE_CSV) -> dict:
    """Compare already-computed per-method MEANS against per-surface/tier maxima.

    No gradient/update modification, no FL rerun, and no target data access.
    """
    with open(source, newline="", encoding="utf-8-sig") as fp:
        rows = list(csv.DictReader(fp))
    if len(rows) != 12:
        raise ValueError("Expected exactly 12 method/surface L2 summary rows")
    seen = set()
    maxima = {t: {s: -math.inf for s in SURFACE_MEASUREMENT}
              for t in ("weak", "strong")}
    annotated = []
    for row in rows:
        _, tier = method_tier(row["method"])
        surface = row["surface"]
        if surface not in SURFACE_MEASUREMENT:
            raise ValueError("Unsupported L2 observation surface: " + surface)
        if row["measurement_level"] != SURFACE_MEASUREMENT[surface]:
            raise ValueError("Exact-surface/proxy classification mismatch")
        if row["population"] != "shadow_val" or int(row["n_runs"]) != 1 or int(row["n_updates"]) != 96:
            raise ValueError("Unsupported summary provenance or aggregation level")
        key = (row["method"], surface)
        if key in seen:
            raise ValueError("Duplicate method/surface row: " + repr(key))
        seen.add(key)
        observed = float(row["mean_relative_L2"])
        if not math.isfinite(observed) or observed < 0:
            raise ValueError("Invalid aggregate relative L2 value")
        maxima[tier][surface] = max(maxima[tier][surface], observed)
        limit = threshold(surface, tier)
        annotated.append({
            "method": row["method"], "tier": tier, "surface": surface,
            "measurement_level": row["measurement_level"],
            "observed_mean_relative_L2": observed,
            "reference_maximum_of_means": limit,
            "within_reference_maximum_of_means": observed <= limit + 1e-12,
        })
    if any(abs(maxima[t][s] - threshold(s, t)) > 1e-12
           for t in maxima for s in SURFACE_MEASUREMENT):
        raise ValueError("Spreadsheet maxima and frozen reference thresholds disagree")
    good = all(r["within_reference_maximum_of_means"] for r in annotated)
    return {
        "status": "PASS" if good else "FAIL",
        "purpose": "retrospective_descriptive_mean_reference_only",
        "source_filename": SOURCE_FILENAME,
        "source_xlsx_sha256": SOURCE_XLSX_SHA256,
        "no_training_updates_modified": True,
        "per_update_maxima_known": False,
        "historical_hard_cap_verified": False,
        "full_update_original_space_L2_verified": False,
        "thresholds": MEAN_RELATIVE_L2_THRESHOLDS,
        "rows": annotated,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=REFERENCE_CSV)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_existing_means(args.source)
    content = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(content)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
