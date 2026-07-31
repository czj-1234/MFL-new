#!/usr/bin/env bash
set -euo pipefail

GPU0="${1:-0}"
GPU1="${2:-1}"
ROUNDS=120
SEEDS=(42 43 44 45 46)
JOB_SCRIPT="scripts/acm_revision/run_modality_exclusive_curve.sh"

if [[ "${GPU0}" == "${GPU1}" ]]; then
  echo "GPU IDs must be different." >&2
  exit 2
fi

if [[ ! -f "${JOB_SCRIPT}" ]]; then
  echo "Missing ${JOB_SCRIPT}. Run this script from the repository root." >&2
  exit 2
fi

echo "============================================================"
echo "Server A: two-GPU queue"
echo "GPU ${GPU0}: concentrations 0.5 and 0.8"
echo "GPU ${GPU1}: concentrations 0.6 and 0.9"
echo "Seeds: ${SEEDS[*]} | rounds/job: ${ROUNDS}"
echo "Completed exact 120-round jobs are skipped locally."
echo "============================================================"

run_queue() {
  local gpu="$1"
  shift
  local concentrations=("$@")
  local seed concentration

  for seed in "${SEEDS[@]}"; do
    for concentration in "${concentrations[@]}"; do
      echo "[SERVER A] GPU=${gpu} seed=${seed} concentration=${concentration}"
      bash "${JOB_SCRIPT}" "${gpu}" "${concentration}" "${ROUNDS}" "${seed}"
    done
  done
}

run_queue "${GPU0}" 0.5 0.8 &
PID0=$!
run_queue "${GPU1}" 0.6 0.9 &
PID1=$!

cleanup() {
  kill -TERM "${PID0}" "${PID1}" 2>/dev/null || true
}
trap cleanup INT TERM

STATUS0=0
STATUS1=0
wait "${PID0}" || STATUS0=$?
wait "${PID1}" || STATUS1=$?
trap - INT TERM

if (( STATUS0 != 0 || STATUS1 != 0 )); then
  echo "[FAIL] Server A queues exited with statuses ${STATUS0}/${STATUS1}." >&2
  exit 1
fi

echo "[DONE] Server A completed its assigned jobs."
