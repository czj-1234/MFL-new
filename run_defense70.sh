#!/usr/bin/env bash
set -euo pipefail

# Auto-detect MFL-new repository root whether this script is placed in:
#   - MFL-new/
#   - MFL-new/results/acm_revision/
#   - another subdirectory inside MFL-new/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_DIR="$(pwd)"

find_repo_root() {
  local d
  for d in \
    "$SCRIPT_DIR" \
    "$START_DIR" \
    "$SCRIPT_DIR/.." \
    "$SCRIPT_DIR/../.." \
    "$SCRIPT_DIR/../../.." \
    "$START_DIR/.." \
    "$START_DIR/../.." \
    "$START_DIR/../../.."
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
  echo "[ERROR] Could not locate the MFL-new repository root."
  echo "Script directory: $SCRIPT_DIR"
  echo "Current directory: $START_DIR"
  echo "Expected a directory containing both:"
  echo "  src/acm_revision"
  echo "  results/acm_revision"
  exit 1
fi

cd "$REPO_ROOT"
echo "[INFO] Repository root detected: $REPO_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

INPUT_ROOT="results/acm_revision/defense70_r150"
OUTPUT_ROOT="results/acm_revision/defense70_postprocess_r150"
LOG_DIR="logs/acm_revision/postprocess"
LOG="${LOG_DIR}/defense70_defenseaware_postprocess.log"

if [[ ! -d "$INPUT_ROOT" ]]; then
  echo "[ERROR] Defense70 result directory not found: $REPO_ROOT/$INPUT_ROOT"
  exit 1
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

python -u - <<'PY' 2>&1 | tee "$LOG"
from __future__ import annotations
import json, shutil, traceback
from pathlib import Path
import numpy as np
import pandas as pd

from src.acm_revision.attacks import read_metadata
from src.acm_revision.core72_postprocess import (
    bootstrap, discover, strict_per_target, task_rows, validate_matrix,
)
from src.acm_revision.privacy_postprocess import _available

INPUT_ROOT = Path("results/acm_revision/defense70_r150")
OUTPUT_ROOT = Path("results/acm_revision/defense70_postprocess_r150")
BASELINE_CELL = Path(
    "results/acm_revision/core72_modality_first_v2_postprocess_r150/"
    "modality_exclusive__c0p7__r150"
)

DEFENSES = [
    "clipgauss_strong", "clipgauss_weak",
    "pca_strong", "pca_weak",
    "proposed_head_ablation", "proposed_single_fullbasis",
    "proposed_system_strong", "proposed_system_weak",
    "random_strong", "random_weak",
]
GROUPS = [
    "classifier_head", "fusion", "image_encoder",
    "text_encoder", "all_shared", "full_update",
]
SEEDS = {"shadow_train":[142], "shadow_val":[242], "target":[42,43,44,45,46]}
SETTING = "modality_exclusive"
CONCENTRATION = 0.7
ROUNDS = 150
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

def successful(df):
    if "error" not in df.columns:
        return df
    return df[df["error"].isna() | (df["error"].astype(str).str.strip()=="")].copy()

def utility_summary(defense, task):
    rows=[]
    for metric in [
        "best_test_acc","best_test_macro_f1","best_test_auroc",
        "final_test_acc","final_test_macro_f1","final_test_auroc"
    ]:
        if metric not in task.columns:
            continue
        vals=pd.to_numeric(task[metric], errors="coerce").dropna()
        if vals.empty:
            continue
        s=bootstrap(vals.tolist())
        rows.append({"defense":defense,"metric":metric,**s})
    return rows

def validate_cell(cell):
    failures=[]
    task_path=cell/"task_metrics_per_target_run.csv"
    if not task_path.exists():
        failures.append("task_metrics_per_target_run.csv missing")
    else:
        task=pd.read_csv(task_path)
        got=sorted(pd.to_numeric(task["target_seed"],errors="coerce").dropna().astype(int).unique())
        if got != SEEDS["target"]:
            failures.append(f"target task seeds mismatch: {got}")

    for group in GROUPS:
        p=cell/f"strict_independent_runs__{group}.csv"
        s=cell/f"strict_independent_summary__{group}.csv"
        if not p.exists():
            failures.append(f"{group}: strict runs missing")
            continue
        if not s.exists():
            failures.append(f"{group}: strict summary missing")
            continue
        df=successful(pd.read_csv(p))
        for protocol in ["cross_partition","cross_partition_temporal"]:
            for model in ["logistic_regression","random_forest","mlp"]:
                q=df[
                    (df["protocol"]==protocol) &
                    (df["attack_model"]==model) &
                    (df["target"]=="dominant_label")
                ]
                got=sorted(pd.to_numeric(q["target_seed"],errors="coerce").dropna().astype(int).unique())
                if got != SEEDS["target"]:
                    failures.append(f"{group}:{protocol}:{model}: seeds={got}")

    report={"status":"PASS" if not failures else "FAIL","failures":failures}
    (cell/"defense_attack_validation.json").write_text(
        json.dumps(report,indent=2),encoding="utf-8"
    )
    marker=cell/"DEFENSE_POSTPROCESS_PASS"
    if failures:
        marker.unlink(missing_ok=True)
        raise RuntimeError("; ".join(failures))
    marker.write_text("PASS\n",encoding="utf-8")

all_attack=[]
all_utility=[]
failures=[]

for i,defense in enumerate(DEFENSES,1):
    src=INPUT_ROOT/defense
    cell=OUTPUT_ROOT/defense/"modality_exclusive__c0p7__r150"
    marker=cell/"DEFENSE_POSTPROCESS_PASS"

    print("\n"+"="*78)
    print(f"[{i}/{len(DEFENSES)}] {defense}")
    print("="*78)

    if marker.exists():
        print("[SKIP] Existing validated PASS output.")
    else:
        try:
            if not src.is_dir():
                raise FileNotFoundError(src)
            if cell.exists():
                print(f"[CLEAN] Removing partial derived output: {cell}")
                shutil.rmtree(cell)
            cell.mkdir(parents=True,exist_ok=True)

            paths=discover(src,SETTING,CONCENTRATION,ROUNDS,SEEDS)
            print(f"[DISCOVER] validated FL runs: {len(paths)}")
            if len(paths)!=7:
                raise RuntimeError(f"expected 7 FL runs, found {len(paths)}")

            frame=read_metadata(paths)
            matrix=validate_matrix(frame,SETTING,CONCENTRATION,SEEDS)
            (cell/"defense_matrix_validation.json").write_text(
                json.dumps(matrix,indent=2),encoding="utf-8"
            )
            task_rows(paths,cell)

            for group in GROUPS:
                if _available(frame,group).empty:
                    raise RuntimeError(f"required attack group unavailable: {group}")
                print(f"[ATTACK] {defense} :: {group}")
                strict_per_target(frame,group,SEEDS["target"],cell,smoke=False)

            (cell/"defense_postprocess_report.json").write_text(
                json.dumps({
                    "status":"PASS","defense":defense,"setting":SETTING,
                    "concentration":CONCENTRATION,"rounds":ROUNDS,
                    "seeds":SEEDS,"groups":GROUPS,
                    "protocols":["cross_partition","cross_partition_temporal"],
                    "attackers":["logistic_regression","random_forest","mlp"],
                    "attacker_training":"defense-aware; observation=observed"
                },indent=2),encoding="utf-8"
            )
            validate_cell(cell)
            print("[PASS]")

        except Exception:
            traceback.print_exc()
            (OUTPUT_ROOT/f"{defense}.failed.txt").write_text(
                traceback.format_exc(),encoding="utf-8"
            )
            failures.append(defense)
            continue

    try:
        task=pd.read_csv(cell/"task_metrics_per_target_run.csv")
        all_utility.extend(utility_summary(defense,task))
        for group in GROUPS:
            q=successful(pd.read_csv(cell/f"strict_independent_summary__{group}.csv"))
            if not q.empty:
                q.insert(0,"defense",defense)
                all_attack.append(q)
    except Exception as e:
        print(f"[WARN] aggregate {defense}: {e}")

if BASELINE_CELL.is_dir():
    try:
        task=pd.read_csv(BASELINE_CELL/"task_metrics_per_target_run.csv")
        all_utility.extend(utility_summary("no_defense",task))
        for group in GROUPS:
            p=BASELINE_CELL/f"strict_independent_summary__{group}.csv"
            if p.exists():
                q=successful(pd.read_csv(p))
                if not q.empty:
                    q.insert(0,"defense","no_defense")
                    all_attack.append(q)
        print("[BASELINE] Added no-defense Core72 c=0.7 baseline.")
    except Exception as e:
        print(f"[WARN] baseline aggregation: {e}")

attack_df=pd.concat(all_attack,ignore_index=True) if all_attack else pd.DataFrame()
utility_df=pd.DataFrame(all_utility)
attack_df.to_csv(OUTPUT_ROOT/"defense70_attack_summary.csv",index=False)
utility_df.to_csv(OUTPUT_ROOT/"defense70_task_utility_summary.csv",index=False)

if not attack_df.empty and not utility_df.empty:
    wide=utility_df.pivot_table(
        index="defense",columns="metric",
        values=["mean","ci_low","ci_high"],aggfunc="first"
    )
    wide.columns=["utility_"+"_".join(map(str,c)) for c in wide.columns]
    wide=wide.reset_index()
    pu=attack_df.merge(wide,on="defense",how="left")
else:
    pu=attack_df.copy()
pu.to_csv(OUTPUT_ROOT/"defense70_privacy_utility.csv",index=False)

completion={
    "status":"PASS" if not failures else "PARTIAL",
    "completed_defenses":[
        d for d in DEFENSES
        if (OUTPUT_ROOT/d/"modality_exclusive__c0p7__r150"/"DEFENSE_POSTPROCESS_PASS").exists()
    ],
    "failed_defenses":failures,
    "expected_defenses":DEFENSES,
}
(OUTPUT_ROOT/"defense70_postprocess_completion.json").write_text(
    json.dumps(completion,indent=2),encoding="utf-8"
)
print(json.dumps(completion,indent=2))
if failures:
    raise SystemExit(2)
PY

echo
echo "=================================================="
echo "[DONE] Defense70 defense-aware postprocess finished"
echo "Output: $REPO_ROOT/$OUTPUT_ROOT"
echo "Log:    $REPO_ROOT/$LOG"
echo "=================================================="
