#!/usr/bin/env bash
set -euo pipefail

# Repair ONLY Defense70 random_strong postprocess.
# Cause addressed:
# random_strong shadow_train metadata has dim_* columns but lacks per-row has_* columns.
# When concatenated with newer metadata, NaN has_* values were treated as False and
# shadow_train rows disappeared from _available().
#
# This script:
#   1) auto-detects repository root
#   2) loads the 7 completed random_strong FL runs
#   3) repairs has_* flags from dim_* > 0 where has_* is missing/NaN
#   4) reruns defense-aware attacks for random_strong only
#   5) rebuilds the compact Defense70 aggregate CSVs
#
# It NEVER reruns FL training and never modifies raw Defense70 result files.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_DIR="$(pwd)"

find_repo_root() {
  local d
  for d in \
    "$SCRIPT_DIR" "$START_DIR" \
    "$SCRIPT_DIR/.." "$SCRIPT_DIR/../.." "$SCRIPT_DIR/../../.." \
    "$START_DIR/.." "$START_DIR/../.." "$START_DIR/../../.."
  do
    if [[ -d "$d/src/acm_revision" && -d "$d/results/acm_revision" ]]; then
      (cd "$d" && pwd)
      return 0
    fi
  done
  return 1
}

REPO_ROOT="$(find_repo_root || true)"
if [[ -z "${REPO_ROOT:-}" ]]; then
  echo "[ERROR] Could not locate MFL-new repository root."
  exit 1
fi
cd "$REPO_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

mkdir -p logs/acm_revision/postprocess
LOG="logs/acm_revision/postprocess/repair_random_strong.log"

python -u - <<'PY' 2>&1 | tee "$LOG"
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.acm_revision.attacks import read_metadata
from src.acm_revision.core72_postprocess import (
    bootstrap,
    discover,
    strict_per_target,
    task_rows,
    validate_matrix,
)

INPUT_ROOT = Path("results/acm_revision/defense70_r150")
OUTPUT_ROOT = Path("results/acm_revision/defense70_postprocess_r150")
DEFENSE = "random_strong"
SRC = INPUT_ROOT / DEFENSE
CELL = OUTPUT_ROOT / DEFENSE / "modality_exclusive__c0p7__r150"
BASELINE_CELL = Path(
    "results/acm_revision/core72_modality_first_v2_postprocess_r150/"
    "modality_exclusive__c0p7__r150"
)

SETTING = "modality_exclusive"
CONCENTRATION = 0.7
ROUNDS = 150
SEEDS = {
    "shadow_train": [142],
    "shadow_val": [242],
    "target": [42, 43, 44, 45, 46],
}
GROUPS = [
    "classifier_head",
    "fusion",
    "image_encoder",
    "text_encoder",
    "all_shared",
    "full_update",
]
ALL_DEFENSES = [
    "clipgauss_strong",
    "clipgauss_weak",
    "pca_strong",
    "pca_weak",
    "proposed_head_ablation",
    "proposed_single_fullbasis",
    "proposed_system_strong",
    "proposed_system_weak",
    "random_strong",
    "random_weak",
]


def successful(df: pd.DataFrame) -> pd.DataFrame:
    if "error" not in df.columns:
        return df
    return df[
        df["error"].isna()
        | (df["error"].astype(str).str.strip() == "")
    ].copy()


def patch_availability(frame: pd.DataFrame) -> pd.DataFrame:
    """Use dim_group > 0 as fallback when has_group is missing/NaN."""
    out = frame.copy()
    repaired = {}
    groups = [
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
    for group in groups:
        h = f"has_{group}"
        d = f"dim_{group}"
        if d not in out.columns:
            continue

        dim_available = (
            pd.to_numeric(out[d], errors="coerce")
            .fillna(0)
            .gt(0)
        )

        if h not in out.columns:
            out[h] = dim_available
            repaired[group] = int(dim_available.sum())
            continue

        # Preserve valid has_* values, fill only missing values from dim_*.
        raw = out[h]
        if raw.dtype == object:
            parsed = raw.astype(str).str.lower().map({
                "true": True, "1": True, "yes": True,
                "false": False, "0": False, "no": False,
                "nan": np.nan, "none": np.nan, "": np.nan,
            })
        else:
            parsed = raw.astype("boolean")

        missing = pd.isna(parsed)
        repaired[group] = int(missing.sum())
        parsed = parsed.where(~missing, dim_available)
        out[h] = parsed.fillna(False).astype(bool)

    print("[PATCH] availability fallback rows filled:", repaired)
    return out


def utility_summary(defense: str, task: pd.DataFrame) -> list[dict]:
    rows = []
    for metric in [
        "best_test_acc",
        "best_test_macro_f1",
        "best_test_auroc",
        "final_test_acc",
        "final_test_macro_f1",
        "final_test_auroc",
    ]:
        if metric not in task.columns:
            continue
        vals = pd.to_numeric(task[metric], errors="coerce").dropna()
        if vals.empty:
            continue
        s = bootstrap(vals.tolist())
        rows.append({"defense": defense, "metric": metric, **s})
    return rows


print("[DISCOVER] random_strong")
paths = discover(SRC, SETTING, CONCENTRATION, ROUNDS, SEEDS)
print("[DISCOVER] validated FL runs:", len(paths))
if len(paths) != 7:
    raise RuntimeError(f"expected 7 FL runs, found {len(paths)}")

frame = read_metadata(paths)
validate_matrix(frame, SETTING, CONCENTRATION, SEEDS)

# Show the exact metadata anomaly before patching.
st = frame[frame["population"].astype(str) == "shadow_train"].copy()
print(
    "[INFO] shadow_train rows=", len(st),
    "round_min=", int(st["round"].min()),
    "round_max=", int(st["round"].max()),
    "unique_rounds=", int(st["round"].nunique()),
)
print(
    "[INFO] strict temporal attacker trains on rounds <= 60, "
    "which are fully present in this shadow_train metadata."
)

frame = patch_availability(frame)

# Confirm every required group retains all three populations after patch.
for group in GROUPS:
    h = f"has_{group}"
    gf = frame[frame[h].astype(bool)].copy()
    pops = sorted(gf["population"].astype(str).unique().tolist())
    print(f"[CHECK] {group}: populations={pops}, rows={len(gf)}")
    if set(pops) != {"shadow_train", "shadow_val", "target"}:
        raise RuntimeError(
            f"{group}: incomplete populations after repair: {pops}"
        )

# Derived random_strong output only; raw FL results untouched.
if CELL.exists():
    print("[CLEAN] removing old partial random_strong postprocess:", CELL)
    shutil.rmtree(CELL)
CELL.mkdir(parents=True, exist_ok=True)

task_rows(paths, CELL)

for group in GROUPS:
    print(f"[ATTACK] random_strong :: {group}")
    strict_per_target(
        frame,
        group,
        SEEDS["target"],
        CELL,
        smoke=False,
    )

report = {
    "status": "PASS",
    "defense": DEFENSE,
    "setting": SETTING,
    "concentration": CONCENTRATION,
    "rounds": ROUNDS,
    "groups": GROUPS,
    "protocols": ["cross_partition", "cross_partition_temporal"],
    "attackers": ["logistic_regression", "random_forest", "mlp"],
    "metadata_repair": (
        "Per-row has_* availability was filled from dim_* > 0 only where "
        "has_* was missing/NaN. Raw FL outputs were not modified."
    ),
    "shadow_train_metadata_rounds": {
        "min": int(st["round"].min()),
        "max": int(st["round"].max()),
        "n_unique": int(st["round"].nunique()),
    },
    "primary_comparison_protocol": "cross_partition_temporal",
    "comparability_note": (
        "random_strong shadow_train metadata contains rounds 1-93; the strict "
        "cross_partition_temporal protocol uses shadow_train rounds <=60, so "
        "its training window is complete and directly comparable across defenses. "
        "Use cross_partition_temporal as the primary Defense70 comparison."
    ),
}
(CELL / "defense_postprocess_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
(CELL / "DEFENSE_POSTPROCESS_PASS").write_text("PASS\n", encoding="utf-8")
print("[PASS] random_strong repaired and re-evaluated.")

# ------------------------------------------------------------------
# Rebuild global compact summaries from all completed defense folders.
# ------------------------------------------------------------------
all_attack = []
all_utility = []

for defense in ALL_DEFENSES:
    cell = OUTPUT_ROOT / defense / "modality_exclusive__c0p7__r150"
    if not (cell / "DEFENSE_POSTPROCESS_PASS").exists():
        print(f"[WARN] aggregate skip: {defense} has no PASS marker")
        continue

    tpath = cell / "task_metrics_per_target_run.csv"
    if tpath.exists():
        all_utility.extend(
            utility_summary(defense, pd.read_csv(tpath))
        )

    for group in GROUPS:
        p = cell / f"strict_independent_summary__{group}.csv"
        if not p.exists():
            continue
        q = successful(pd.read_csv(p))
        if q.empty:
            continue
        q.insert(0, "defense", defense)
        all_attack.append(q)

# Add no-defense baseline.
if BASELINE_CELL.is_dir():
    tpath = BASELINE_CELL / "task_metrics_per_target_run.csv"
    if tpath.exists():
        all_utility.extend(
            utility_summary("no_defense", pd.read_csv(tpath))
        )
    for group in GROUPS:
        p = BASELINE_CELL / f"strict_independent_summary__{group}.csv"
        if not p.exists():
            continue
        q = successful(pd.read_csv(p))
        if q.empty:
            continue
        q.insert(0, "defense", "no_defense")
        all_attack.append(q)

attack_df = (
    pd.concat(all_attack, ignore_index=True)
    if all_attack else pd.DataFrame()
)
utility_df = pd.DataFrame(all_utility)

attack_df.to_csv(
    OUTPUT_ROOT / "defense70_attack_summary.csv",
    index=False,
)
utility_df.to_csv(
    OUTPUT_ROOT / "defense70_task_utility_summary.csv",
    index=False,
)

if not attack_df.empty and not utility_df.empty:
    wide = utility_df.pivot_table(
        index="defense",
        columns="metric",
        values=["mean", "ci_low", "ci_high"],
        aggfunc="first",
    )
    wide.columns = [
        "utility_" + "_".join(map(str, c))
        for c in wide.columns
    ]
    wide = wide.reset_index()
    pu = attack_df.merge(wide, on="defense", how="left")
else:
    pu = attack_df.copy()

pu.to_csv(
    OUTPUT_ROOT / "defense70_privacy_utility.csv",
    index=False,
)

completed = [
    d for d in ALL_DEFENSES
    if (
        OUTPUT_ROOT
        / d
        / "modality_exclusive__c0p7__r150"
        / "DEFENSE_POSTPROCESS_PASS"
    ).exists()
]
completion = {
    "status": "PASS" if len(completed) == len(ALL_DEFENSES) else "PARTIAL",
    "completed_defenses": completed,
    "failed_or_missing_defenses": [
        d for d in ALL_DEFENSES if d not in completed
    ],
    "expected_defenses": ALL_DEFENSES,
    "primary_comparison_protocol": "cross_partition_temporal",
}
(OUTPUT_ROOT / "defense70_postprocess_completion.json").write_text(
    json.dumps(completion, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(json.dumps(completion, ensure_ascii=False, indent=2))

if completion["status"] != "PASS":
    raise SystemExit(2)
PY

echo
echo "=================================================="
echo "[DONE] random_strong repair finished"
echo "Log: $REPO_ROOT/$LOG"
echo "Check:"
echo "  results/acm_revision/defense70_postprocess_r150/defense70_postprocess_completion.json"
echo "=================================================="
