#!/usr/bin/env bash
# Server A: eight c=.7 uncentered-contrast ASLWF runs, four on each GPU.
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ "$(git branch --show-current)" != "aslwf-supplement16-20261022" ]]; then
  echo "[FATAL] checkout aslwf-supplement16-20261022 first" >&2; exit 2
fi
command -v nvidia-smi >/dev/null || { echo "[FATAL] GPU not available" >&2; exit 2; }
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU0:-0}"
mkdir -p logs/acm_revision/supplement16/uncentered
echo "[PREP] compute shadow-only uncentered subspaces (head stays locked)"
python -m src.acm_revision.supplement16 prepare-uncentered \
  2>&1 | tee logs/acm_revision/supplement16/uncentered/shadow_basis_preparation.log
unset CUDA_VISIBLE_DEVICES
python -m src.acm_revision.supplement16 generate --phase uncentered
PHASE=uncentered

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
if [[ "$GPU0" == "$GPU1" ]]; then
  echo "[FATAL] GPU0 and GPU1 must be different physical GPU identifiers" >&2
  exit 2
fi
command -v nvidia-smi >/dev/null || { echo "[FATAL] NVIDIA GPU driver unavailable" >&2; exit 2; }
command -v flock >/dev/null || { echo "[FATAL] flock required" >&2; exit 2; }
for gpu in "$GPU0" "$GPU1"; do
  nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader || exit 2
done
export PYTHONUNBUFFERED=1
export DEFENSE70_MIN_FREE_MIB="${MIN_FREE_MIB:-23000}"
export DEFENSE70_CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
export DEFENSE70_CHECKPOINT_KEEP_LAST="${CHECKPOINT_KEEP_LAST:-2}"
export DEFENSE70_MAX_RETRIES="${MAX_RETRIES:-2}"
export DEFENSE70_STARTUP_WAIT="${STARTUP_WAIT:-90}"
phase="$PHASE"
export DEFENSE70_LOG_ROOT="logs/acm_revision/supplement16/$phase"
export DEFENSE70_STATE_ROOT="logs/acm_revision/supplement16/$phase/state"
export DEFENSE70_CHECKPOINT_ROOT="checkpoints/acm_revision/supplement16/$phase"
export DEFENSE70_RUNTIME_CFG_ROOT="logs/acm_revision/supplement16/$phase/runtime_configs"
mkdir -p "$DEFENSE70_LOG_ROOT"
plan_dir="configs/acm_revision/generated/supplement16/$phase"
run_queue() {
  local gpu_id="$1" queue_id="$2"
  local queue="$plan_dir/gpu${queue_id}.tsv"
  [[ -s "$queue" ]] || { echo "[ERROR] queue missing: $queue" >&2; return 2; }
  local done=0
  while IFS=$'\t' read -r job cfg population seed; do
    [[ "$job" == job_id || -z "${job:-}" ]] && continue
    [[ -f "$cfg" ]] || { echo "[ERROR] $cfg missing" >&2; return 3; }
    echo "[$(date -Iseconds)] [GPU $gpu_id] start $job / $population / seed $seed"
    bash scripts/acm_revision/run_defense70_job.sh "$gpu_id" "$job" "$cfg" || return $?
    done=$((done + 1))
  done < "$queue"
  [[ "$done" -eq 4 ]] || { echo "[FATAL] queue $queue had $done jobs, expected 4" >&2; return 5; }
}
run_queue "$GPU0" 0 >"$DEFENSE70_LOG_ROOT/gpu0.queue.log" 2>&1 &
pid0=$!
run_queue "$GPU1" 1 >"$DEFENSE70_LOG_ROOT/gpu1.queue.log" 2>&1 &
pid1=$!
set +e
wait "$pid0"; code0=$?
wait "$pid1"; code1=$?
set -e
if [[ "$code0" -ne 0 || "$code1" -ne 0 ]]; then
  echo "[FAILED] queue exits GPU0=$code0 GPU1=$code1" >&2
  echo "Inspect $DEFENSE70_LOG_ROOT/gpu{0,1}.queue.log" >&2
  exit 1
fi
python -m src.acm_revision.supplement16 verify --phase "$phase"
# End-to-end nine-surface, CP/CPT, LR/RF/MLP defense-aware audit.
concentration=0.7
[[ "$phase" == "c09" ]] && concentration=0.9
audit_args=(
  --root "results/acm_revision/supplement16/$phase"
  --out "results/acm_revision/supplement16/audits/${phase}_attacks"
  --concentration "$concentration"
)
if [[ -n "${BASELINE_TASK_CSV:-}" ]]; then
  [[ -f "$BASELINE_TASK_CSV" ]] || { echo "[FATAL] baseline CSV missing: $BASELINE_TASK_CSV" >&2; exit 3; }
  audit_args+=(--baseline-task-csv "$BASELINE_TASK_CSV")
fi
python scripts/acm_revision/audit_headfix.py "${audit_args[@]}" \
  >"$DEFENSE70_LOG_ROOT/all_surface_CP_CPT_audit.log" 2>&1
echo "[COMPLETE] $phase: eight runs + nine-surface CP/CPT audit verified"

