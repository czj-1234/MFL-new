#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d "src" || ! -d "configs" ]]; then
  echo "[ERROR] Run this script from the MFL-new project root."
  exit 1
fi

mkdir -p logs

BASIS="results/defense_seed42_assoc07/subspace/final_r5.npz"

[[ -f "$BASIS" ]] || { echo "[ERROR] Missing $BASIS"; exit 1; }

echo "========================================"
echo "Running Random baseline: rank=5, alpha=1.0 on GPU 1"
echo "Basis: $BASIS"
echo "Log: logs/defense_random_r5_a10_gpu1.log"
echo "========================================"

CUDA_VISIBLE_DEVICES=1 python -u -m src.scripts.run_defense_comparison \
  --method random \
  --basis "$BASIS" \
  --rank 5 \
  --alpha 1.0 \
  2>&1 | tee logs/defense_random_r5_a10_gpu1.log

echo "========================================"
echo "Completed Random baseline: rank=5, alpha=1.0 on GPU 1"
echo "========================================"
