#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-1}"
ROUNDS=150
SEEDS=(42 43 44 45 46)
CONCENTRATIONS=(0.7 1.0)
JOB_SCRIPT="scripts/acm_revision/run_modality_exclusive_curve.sh"

if [[ ! -f "${JOB_SCRIPT}" ]]; then
  echo "Missing ${JOB_SCRIPT}. Run this script from the repository root." >&2
  exit 2
fi

echo "============================================================"
echo "Server B: single-GPU queue"
echo "GPU ${GPU}: concentrations ${CONCENTRATIONS[*]}"
echo "Seeds: ${SEEDS[*]} | rounds/job: ${ROUNDS}"
echo "Completed exact 150-round jobs are skipped locally."
echo "Older 30/120-round results are kept separately and will not be treated as complete."
echo "============================================================"

for seed in "${SEEDS[@]}"; do
  for concentration in "${CONCENTRATIONS[@]}"; do
    echo "[SERVER B] GPU=${GPU} seed=${seed} concentration=${concentration} rounds=${ROUNDS}"
    bash "${JOB_SCRIPT}" "${GPU}" "${concentration}" "${ROUNDS}" "${seed}"
  done
done

echo "[DONE] Server B completed its assigned 150-round jobs."
