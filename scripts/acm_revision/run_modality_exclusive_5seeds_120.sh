#!/usr/bin/env bash
set -uo pipefail

GPU0="${1:-0}"
GPU1="${2:-1}"
ROUNDS=120
SEEDS=(42 43 44 45 46)
CONCENTRATIONS=(0.5 0.6 0.7 0.8 0.9 1.0)
JOB_SCRIPT="scripts/acm_revision/run_modality_exclusive_curve.sh"

if [[ "${GPU0}" == "${GPU1}" ]]; then
  echo "GPU0 and GPU1 must be different device IDs." >&2
  exit 2
fi

if [[ ! -f "${JOB_SCRIPT}" ]]; then
  echo "Missing ${JOB_SCRIPT}. Run this script from the repository root." >&2
  exit 2
fi

echo "============================================================"
echo "Five-seed modality-exclusive utility sweep"
echo "Seeds: ${SEEDS[*]}"
echo "Concentrations: ${CONCENTRATIONS[*]}"
echo "Rounds/job: ${ROUNDS}"
echo "Configuration: 12 clients x 200 samples, batch=16, local_epochs=1, FedAvg"
echo "GPU assignment per seed:"
echo "  GPU ${GPU0}: 0.5, 0.7, 0.9"
echo "  GPU ${GPU1}: 0.6, 0.8, 1.0"
echo "Already completed exact 120-round jobs are skipped automatically."
echo "============================================================"

ACTIVE_PIDS=()
cleanup() {
  for pid in "${ACTIVE_PIDS[@]:-}"; do
    kill -TERM "${pid}" 2>/dev/null || true
  done
}
trap cleanup INT TERM

run_pair() {
  local seed="$1"
  local conc0="$2"
  local conc1="$3"
  local pid0 pid1 status0=0 status1=0

  echo "------------------------------------------------------------"
  echo "Seed ${seed}: GPU ${GPU0} -> c=${conc0}; GPU ${GPU1} -> c=${conc1}"
  echo "------------------------------------------------------------"

  bash "${JOB_SCRIPT}" "${GPU0}" "${conc0}" "${ROUNDS}" "${seed}" &
  pid0=$!
  bash "${JOB_SCRIPT}" "${GPU1}" "${conc1}" "${ROUNDS}" "${seed}" &
  pid1=$!
  ACTIVE_PIDS=("${pid0}" "${pid1}")

  wait "${pid0}" || status0=$?
  wait "${pid1}" || status1=$?
  ACTIVE_PIDS=()

  if (( status0 != 0 || status1 != 0 )); then
    echo "[FAIL] seed=${seed}, pair c=${conc0}/${conc1}; statuses=${status0}/${status1}" >&2
    exit 1
  fi
}

for seed in "${SEEDS[@]}"; do
  # A seed barrier is intentional: all six concentrations for one seed finish
  # before the next independent seed starts.
  run_pair "${seed}" 0.5 0.6
  run_pair "${seed}" 0.7 0.8
  run_pair "${seed}" 0.9 1.0
  echo "[SEED DONE] ${seed}"
done

trap - INT TERM

echo "============================================================"
echo "[ALL DONE] seeds 42-46, concentrations 0.5-1.0, 120 rounds"
echo "Exact jobs that already had summary.json were not rerun."
echo "============================================================"
