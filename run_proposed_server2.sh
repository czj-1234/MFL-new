#!/usr/bin/env bash
set -euo pipefail

# Run this script from the MFL-new project root.
if [[ ! -d "src" || ! -d "configs" ]]; then
  echo "[ERROR] Please run this script from the MFL-new project root."
  exit 1
fi

mkdir -p logs

PYTHON_BIN="${PYTHON_BIN:-python}"
BASIS="results/defense_seed42_assoc07/subspace/final_r5.npz"
SHADOW="results/defense_seed42_assoc07/subspace/combined/associated_all.npz"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] Python command not found: ${PYTHON_BIN}"
  echo "Activate the mfl environment first, or run with:"
  echo "  PYTHON_BIN=/full/path/to/python ./run_proposed_server2.sh"
  exit 1
fi

[[ -f "${BASIS}" ]] || { echo "[ERROR] Missing basis file: ${BASIS}"; exit 1; }
[[ -f "${SHADOW}" ]] || { echo "[ERROR] Missing shadow updates: ${SHADOW}"; exit 1; }

run_one () {
  local rank="$1"
  local alpha="$2"
  local log_file="$3"

  echo "============================================================"
  echo "Running Proposed: rank=${rank}, alpha=${alpha}"
  echo "Log: ${log_file}"
  echo "Started at: $(date)"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" -u     -m src.scripts.run_defense_comparison     --method proposed     --basis "${BASIS}"     --shadow-updates "${SHADOW}"     --rank "${rank}"     --alpha "${alpha}"     2>&1 | tee -a "${log_file}"

  echo "Completed rank=${rank}, alpha=${alpha} at $(date)"
}


run_one 3 1.00 "logs/proposed_r3_a100.log"
run_one 5 0.50 "logs/proposed_r5_a050.log"

echo "SERVER 2 EXPERIMENTS COMPLETED"
